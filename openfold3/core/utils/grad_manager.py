# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from collections.abc import Iterable

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torchmetrics import MaxMetric, MeanMetric

from openfold3.core.utils.tensor_utils import tensor_tree_map

logger = logging.getLogger(__name__)

_GRAD_REDUCE_BUCKET_BYTES = 64 * 1024 * 1024
_FLOAT32_BYTES = 4


def _bucket_tensors_by_numel(
    tensors: list[torch.Tensor], max_float32_bytes: int
) -> list[list[torch.Tensor]]:
    """Partition tensors in parameter order for bounded FP32 reductions.

    A tensor larger than the target occupies a bucket by itself and is sliced
    by the reduction loop.
    """

    max_numel = max(1, max_float32_bytes // _FLOAT32_BYTES)
    buckets: list[list[torch.Tensor]] = []
    current: list[torch.Tensor] = []
    current_numel = 0
    for tensor in tensors:
        tensor_numel = tensor.numel()
        if current and current_numel + tensor_numel > max_numel:
            buckets.append(current)
            current = []
            current_numel = 0
        current.append(tensor)
        current_numel += tensor_numel
        if current_numel >= max_numel:
            buckets.append(current)
            current = []
            current_numel = 0
    if current:
        buckets.append(current)
    return buckets


@torch.no_grad()
def compute_global_norm(
    parameters: torch.Tensor | Iterable[torch.Tensor],
) -> [torch.Tensor, list]:
    """
    Calculates the global norm of all parameters that have gradients.
    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients for norm calculation.
    Returns:
        global_norm (torch.Tensor): The scalar global norm.
        params_with_grad (list): The list of parameters that have gradients.
    """
    params_with_grad = [p for p in parameters if p.grad is not None]

    if not params_with_grad:
        device = next(iter(parameters)).device
        return torch.tensor(0.0, device=device), []

    # Calculate the total norm of all parameter gradients
    per_tensor_norms = [
        torch.linalg.vector_norm(p.grad.float(), ord=2) for p in params_with_grad
    ]

    global_norm = torch.linalg.vector_norm(torch.stack(per_tensor_norms), ord=2)

    return global_norm, params_with_grad


class PerSampleGradManager:
    """
    Manages manual optimization for per-sample gradient clipping and accumulation.
    Manual optimization is required because PyTorch Lightning does not natively support
    per-sample gradient clipping, and instead performs this at the batch level.
    """

    def __init__(
        self,
        gradient_clip_val: int | float | None = None,
        accumulate_grad_batches: int = 1,
        log_grad_norm: bool = False,
    ):
        """
        Args:
            gradient_clip_val (int | float | None): The value to clip the global norm of
                per-sample gradients to
            accumulate_grad_batches (int): Amount of gradient accumulation steps
            log_grad_norm (bool): Whether to log gradient norm metrics
        """
        self.max_grad_norm = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches
        self.log_grad_norm = log_grad_norm

        self.grad_accumulator = {}
        self._params_to_update = {}

        # Track the number of valid samples per parameter for handling
        # of confidence heads
        self.parameter_participation_counts = {}

        # Track the number of accumulated gradients
        self.accum_count = 0

        # Used for logging the average of unclipped per-sample norms
        self.avg_unclipped_norm_metric = MeanMetric() if log_grad_norm else None
        self.max_unclipped_norm_metric = MaxMetric() if log_grad_norm else None

        # Pointers to these objects will be linked in the setup() call
        self._model = None
        self._trainer = None
        self._logger = None
        self._device = None

        # Cache max_norm tensor
        self._max_norm_tensor = None

    def setup(
        self, model: torch.nn.Module, trainer: pl.Trainer, logger: pl.loggers.Logger
    ):
        """
        Initializes the gradient accumulator and links essential components.
        This must be called from the LightningModule's setup() hook.
        """
        self._model = model
        self._trainer = trainer
        self._logger = logger

        self._params_to_update = {
            name: p for name, p in self._model.named_parameters() if p.requires_grad
        }

        self._device = next(iter(self._params_to_update.values())).device

        self.grad_accumulator = {
            name: torch.zeros_like(p, requires_grad=False)
            for name, p in self._params_to_update.items()
        }

        if self.max_grad_norm is not None:
            self._max_norm_tensor = torch.tensor(
                self.max_grad_norm, device=self._device
            )

        if self.log_grad_norm:
            self.avg_unclipped_norm_metric = self.avg_unclipped_norm_metric.to(
                self._device
            )
            self.max_unclipped_norm_metric = self.max_unclipped_norm_metric.to(
                self._device
            )

    @torch.no_grad()
    def _clip_grads(
        self, logging_info: dict | None = None, disabled_params: set | None = None
    ):
        """Clips the gradients currently stored in self._model.parameters()"""
        if disabled_params is None:
            disabled_params = set()

        params_enabled = [
            param
            for name, param in self._params_to_update.items()
            if name not in disabled_params
        ]
        global_norm, params_with_grad = compute_global_norm(parameters=params_enabled)

        if not params_with_grad:
            return

        # Log the metrics even if clipping is disabled
        if self.log_grad_norm:
            self.avg_unclipped_norm_metric.update(global_norm)
            self.max_unclipped_norm_metric.update(global_norm)

        # Skip clipping if it's disabled
        if self.max_grad_norm is None:
            return

        if logging_info is not None and self._trainer.global_step > 1000:
            self.log_outlier_samples(
                logging_info=logging_info, global_norm=global_norm.item()
            )

        # Clip norm and compute rescale factor
        # Note: We use maximum here to avoid CPU <-> GPU synchronization that can
        # occur with additional conditional `if global_norm > self.max_grad_norm`
        clip_coef = self._max_norm_tensor / torch.maximum(
            global_norm, self._max_norm_tensor
        )

        # Rescale gradients
        for p in params_with_grad:
            p.grad.mul_(clip_coef.to(p.dtype))

    @torch.no_grad()
    def _sync_and_average_grads(self):
        """
        Sums gradients across all ranks and averages by the
        total number of accumulated samples globally.

        Uses per-parameter active counts to handle sparse grads correctly.
        The confidence module does not get grads from distillation samples
        and samples with resolution out of range, so their grads will be divided
        by the active number of samples instead.
        """
        # Get effective batch size
        local_count = torch.tensor(self.accum_count, device=self.device)
        global_count = self._trainer.strategy.reduce(
            local_count, reduce_op=dist.ReduceOp.SUM
        ).item()

        self.log_unclipped_grad_metrics(global_count=global_count)

        if global_count == 0:
            # Zero the grads, they might contain stale values from the accumulator
            for p in self._params_to_update.values():
                if p.grad is not None:
                    p.grad.zero_()
            return

        # Collect grads and parameters
        param_names = sorted(self._params_to_update.keys())
        params = [self._params_to_update[n] for n in param_names]
        grads = [p.grad for p in params]

        if not grads:
            return

        # Per-parameter sample counts [N_params]
        local_counts = [
            self.parameter_participation_counts.get(n, 0) for n in param_names
        ]
        global_active_counts = self._trainer.strategy.reduce(
            torch.tensor(local_counts, device=self.device), reduce_op=dist.ReduceOp.SUM
        )

        if int(self._trainer.world_size) == 1:
            # Preserve the qualified single-GPU execution path byte-for-byte;
            # distributed bucketing is intentionally scoped to world_size > 1.
            flat_grad = _flatten_dense_tensors(grads).float()
            reduced_flat = self._trainer.strategy.reduce(
                flat_grad, reduce_op=dist.ReduceOp.SUM
            )
            summed_grads = _unflatten_dense_tensors(reduced_flat, grads)
            for index, (parameter, summed_grad) in enumerate(
                zip(params, summed_grads)
            ):
                count = global_active_counts[index]
                if count > 0:
                    parameter.grad.copy_(summed_grad / count)
                else:
                    parameter.grad.zero_()
            return

        # Reduce fixed-order FP32 buckets.  Write each result back immediately
        # so views do not retain every reduced flat buffer until the final
        # parameter.  Oversized tensors are sliced to preserve the same memory
        # bound.  This changes collective boundaries, so equivalence is
        # numerical rather than cross-protocol bitwise.
        max_numel = _GRAD_REDUCE_BUCKET_BYTES // _FLOAT32_BYTES
        param_offset = 0
        for grad_bucket in _bucket_tensors_by_numel(
            grads, _GRAD_REDUCE_BUCKET_BYTES
        ):
            bucket_size = len(grad_bucket)
            bucket_params = params[param_offset : param_offset + bucket_size]
            bucket_counts = global_active_counts[
                param_offset : param_offset + bucket_size
            ]
            if bucket_size == 1 and grad_bucket[0].numel() > max_numel:
                grad_flat = grad_bucket[0].reshape(-1)
                output_flat = bucket_params[0].grad.reshape(-1)
                count = bucket_counts[0]
                for start in range(0, grad_flat.numel(), max_numel):
                    stop = min(start + max_numel, grad_flat.numel())
                    reduced = self._trainer.strategy.reduce(
                        grad_flat[start:stop].float(),
                        reduce_op=dist.ReduceOp.SUM,
                    )
                    denominator = count.clamp_min(1).to(dtype=reduced.dtype)
                    active = (count > 0).to(dtype=reduced.dtype)
                    output_flat[start:stop].copy_(
                        (reduced / denominator) * active
                    )
                    del reduced
            else:
                flat_grad = _flatten_dense_tensors(grad_bucket).float()
                reduced_flat = self._trainer.strategy.reduce(
                    flat_grad, reduce_op=dist.ReduceOp.SUM
                )
                summed_views = _unflatten_dense_tensors(
                    reduced_flat, grad_bucket
                )
                for p, summed_grad, count in zip(
                    bucket_params, summed_views, bucket_counts
                ):
                    # Keep this branch on device.  Python ``if count > 0``
                    # performs one GPU-to-host sync per parameter.
                    denominator = count.clamp_min(1).to(dtype=summed_grad.dtype)
                    active = (count > 0).to(dtype=summed_grad.dtype)
                    p.grad.copy_((summed_grad / denominator) * active)
                del flat_grad, reduced_flat, summed_views
            param_offset += bucket_size

        if param_offset != len(params):
            raise RuntimeError(
                "gradient bucket traversal did not consume every parameter"
            )

    @torch.no_grad()
    def clip_and_accumulate(
        self,
        logging_info: dict | None = None,
        disabled_params: set | None = None,
    ):
        """
        Clips the current per-sample gradient in self._model.parameters()
        and adds it to the internal gradient accumulator.

        This should be called after self.manual_backward(loss),
        inside of a self.trainer.model.no_sync() context.

        Args:
            logging_info:
                Info for logging outliers.
            disabled_params:
                A set of parameter names for which a sample is not active.
                If None, assumes all parameters are active.
        """
        # Clip the single-sample grads
        self._clip_grads(logging_info=logging_info, disabled_params=disabled_params)

        if disabled_params is None:
            disabled_params = set()

        # Manually accumulate clipped grads and track param participation
        for name, param in self._params_to_update.items():
            if name in disabled_params:
                continue

            if param.grad is not None:
                self.grad_accumulator[name].add_(param.grad)

                if name not in self.parameter_participation_counts:
                    self.parameter_participation_counts[name] = 0

                self.parameter_participation_counts[name] += 1

        # Increment the global counter (still used for logging)
        # TODO: Get rid of this later
        self.accum_count += 1

    @torch.no_grad()
    def sync_and_average_grads(self):
        """
        Prepares the gradients for the optimizer step.
        1. Copies the summed grads from the grad accumulator to
           self._model.parameters().
        2. Syncs and averages grads across all ranks by the
           active number of accumulated samples per parameter.

        This should be called before opt.step().
        """
        # Copy summed grads from accumulator
        for name, param in self._params_to_update.items():
            param.grad = self.grad_accumulator[name].clone()

        # Sync and average globally
        self._sync_and_average_grads()

    @torch.no_grad()
    def reset_accumulator(self):
        """
        Resets the gradient accumulator and counter to zeros.
        This should be called after opt.step().
        """
        for acc_grad in self.grad_accumulator.values():
            acc_grad.zero_()

        # Reset the counters
        self.accum_count = 0
        self.parameter_participation_counts = {}

        # Reset the metric
        if self.log_grad_norm:
            self.avg_unclipped_norm_metric.reset()
            self.max_unclipped_norm_metric.reset()

    @torch.no_grad()
    def log_outlier_samples(
        self,
        logging_info: dict | None,
        global_norm: float,
        warning_norm_multiplier: float = 5.0,
        log_after_step: int = 1000,
    ):
        # TODO: Tune thresholds and make this more informative

        # Only start logging outlier unclipped grads after warmup by default
        warning_threshold = self.max_grad_norm * warning_norm_multiplier
        if (
            logging_info is not None
            and self._trainer.global_step > log_after_step
            and global_norm > warning_threshold
        ):
            pdb_id = logging_info.get("pdb_id")
            preferred_chain_or_interface = logging_info.get(
                "preferred_chain_or_interface"
            )
            logger.warning(
                f"Large gradient norm for {pdb_id} with preferred chain or interface "
                f"{preferred_chain_or_interface} on rank {self._trainer.global_rank} "
                f"step {self._trainer.global_step}: {global_norm}"
            )

    @torch.no_grad()
    def log_unclipped_grad_metrics(self, global_count: int):
        """
        Logs the average and max of the unclipped per-sample gradient norms
        seen during accumulation.
        This should be called after clip_and_accumulate() and before grads are synced.
        """
        if global_count > 0 and self.log_grad_norm:
            avg_per_sample_norm = self.avg_unclipped_norm_metric.compute()
            max_per_sample_norm = self.max_unclipped_norm_metric.compute()

            if self._logger is not None:
                self._logger.log_metrics(
                    {"extra_gradients/avg_unclipped_grad_norm": avg_per_sample_norm},
                    step=self._trainer.global_step,
                )
                self._logger.log_metrics(
                    {"extra_gradients/max_unclipped_grad_norm": max_per_sample_norm},
                    step=self._trainer.global_step,
                )

    @torch.no_grad()
    def log_average_grad_norm(self):
        """
        Calculates and logs the global norm of the final, averaged gradients.
        This should be called after sync_grads() and before optimizer.step().
        """
        if not self.log_grad_norm or self._logger is None:
            return

        global_norm, params_with_grad = compute_global_norm(
            parameters=self._params_to_update.values()
        )

        if not params_with_grad:
            return

        self._logger.log_metrics(
            {"extra_gradients/avg_clipped_grad_norm": global_norm},
            step=self._trainer.global_step,
        )

    @property
    def device(self):
        return self._device

    @torch.no_grad()
    def to(self, device):
        self.grad_accumulator = tensor_tree_map(
            lambda t: t.to(device), self.grad_accumulator
        )
        if self.log_grad_norm:
            self.avg_unclipped_norm_metric = self.avg_unclipped_norm_metric.to(device)
            self.max_unclipped_norm_metric = self.max_unclipped_norm_metric.to(device)

        if self._max_norm_tensor is not None:
            self._max_norm_tensor = self._max_norm_tensor.to(device)

        self._device = device
        return self
