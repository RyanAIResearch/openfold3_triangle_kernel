# Copyright 2026 AlQuraishi Laboratory
# Copyright 2026 Advanced Micro Devices, Inc.
# Copyright 2021 DeepMind Technologies Limited
# Modified by RyanAIResearch in 2026 to integrate TurboPairFormer Triangle Attention.
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

"""Triangle attention layers."""

from functools import partial, partialmethod
from itertools import product

import torch
import torch.nn as nn

import openfold3.core.config.default_linear_init_config as lin_init
from openfold3.core.model.primitives import Attention, LayerNorm, Linear
from openfold3.core.utils.checkpointing import is_inference_without_checkpoint
from openfold3.core.utils.chunk_utils import chunk_layer
from openfold3.core.utils.tensor_utils import (
    permute_final_dims,
)


class TriangleAttention(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        no_heads,
        starting=True,
        inf=1e9,
        linear_init_params=lin_init.tri_att_init,
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Overall hidden channel dimension (not per-head)
            no_heads:
                Number of attention heads
        """
        super().__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf

        self.layer_norm = LayerNorm(self.c_in)

        self.linear_z = Linear(c_in, self.no_heads, **linear_init_params.linear_z)

        self.mha = Attention(
            c_q=self.c_in,
            c_k=self.c_in,
            c_v=self.c_in,
            c_hidden=self.c_hidden,
            no_heads=self.no_heads,
            linear_init_params=linear_init_params.mha,
        )

    @torch.jit.ignore
    def _chunk(
        self,
        x: torch.Tensor,
        biases: list[torch.Tensor],
        chunk_size: int,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_triton_triangle_kernels: bool = False,
        use_stable_triangle_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _allow_fp32_inference: bool = True,
    ) -> torch.Tensor:
        if use_stable_triangle_attention:
            return self._chunk_stable(
                x, biases, chunk_size, inplace_safe, _allow_fp32_inference
            )

        mha_inputs = {
            "q_x": x,
            "kv_x": x,
            "biases": biases,
        }

        return chunk_layer(
            partial(
                self.mha,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                use_triton_triangle_kernels=use_triton_triangle_kernels,
                use_lma=use_lma,
            ),
            mha_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(x.shape[:-2]),
            _out=x if inplace_safe else None,
        )

    def _chunk_stable(
        self,
        x: torch.Tensor,
        biases: list[torch.Tensor],
        chunk_size: int,
        inplace_safe: bool,
        allow_fp32_inference: bool,
    ) -> torch.Tensor:
        """Chunk outer rows while preserving Turbo's shared full pair bias.

        The generic chunker flattens batch and outer rows together, which
        removes Turbo's batch dimension and expands the pair bias across rows.
        Instead, each call keeps [1, outer_chunk, N, C] inputs, the matching
        mask rows and the complete [1, 1, H, N, N] bias for one batch item.
        Only independent outer rows are partitioned; softmax still spans N.
        """
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        batch_shape = x.shape[:-3]
        outer, sequence = x.shape[-3:-1]
        mask_bias = biases[0].expand(*batch_shape, outer, 1, 1, sequence)
        pair_bias = biases[1].expand(*batch_shape, 1, self.no_heads, sequence, sequence)
        output = x if inplace_safe else None
        for batch_index in product(*(range(size) for size in batch_shape)):
            # expand above creates broadcast views, never an outer-row copy
            # of the N x N pair bias. Preserve this sharing for every chunk.
            full_pair_bias = pair_bias[batch_index].unsqueeze(0)
            for start in range(0, outer, chunk_size):
                rows = slice(start, min(start + chunk_size, outer))
                index = batch_index + (rows,)
                input_chunk = x[index].unsqueeze(0)
                output_chunk = self.mha(
                    q_x=input_chunk,
                    kv_x=input_chunk,
                    biases=[mask_bias[index].unsqueeze(0), full_pair_bias],
                    use_stable_triangle_attention=True,
                    _allow_fp32_inference=allow_fp32_inference,
                ).squeeze(0)
                if output is None:
                    # Match the projection's output dtype under autocast.
                    output = output_chunk.new_empty(x.shape)
                output[index] = output_chunk
        if output is None:
            raise ValueError("stable Triangle Attention requires nonempty pair rows")
        return output

    def _fused_fp32_inference(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        transpose_bias: bool,
        chunk_size: int | None,
    ) -> torch.Tensor | None:
        """TurboPairFormer's fused TF32 inference route, or None for the unfused one.

        Taken for FP32 inference with TF32 enabled, unchunked or with outer-row
        chunks. LayerNorm and the five input projections run in fused kernels
        and the gate is applied inside the attention kernel; the result has the
        unfused route's bits. Other configurations return None.
        """
        if (
            x.dtype != torch.float32
            or not x.is_cuda
            or not is_inference_without_checkpoint(self, x, mask)
        ):
            return None
        if (
            torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") != torch.float32
        ):
            return None
        mha = self.mha
        if mha.linear_g is None:
            return None
        projections = (
            mha.linear_q,
            mha.linear_k,
            mha.linear_v,
            mha.linear_g,
            mha.linear_o,
            self.linear_z,
        )
        if any(
            getattr(layer, "precision", None) not in (None, torch.float32)
            for layer in projections
        ):
            return None
        from openfold3.core.kernels.turbopairformer import (
            turbopairformer_attention_fp32 as fp32_attention,
        )

        if mask is None:
            mask = x.new_ones(x.shape[:-1])
        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)
        supported, _ = fp32_attention.is_triangle_attention_module_fp32_supported(
            x,
            mask,
            layer_norm_weight=self.layer_norm.weight,
            layer_norm_bias=self.layer_norm.bias,
            projection_weights=tuple(layer.weight for layer in projections),
            projection_biases=tuple(layer.bias for layer in projections),
            heads=mha.no_heads,
            head_dim=mha.c_hidden,
            chunk_size=chunk_size,
        )
        if not supported:
            return None
        output = fp32_attention.turbopairformer_triangle_attention_module_fp32(
            x,
            mask,
            layer_norm_weight=self.layer_norm.weight,
            layer_norm_bias=self.layer_norm.bias,
            layer_norm_epsilon=self.layer_norm.eps,
            q_weight=mha.linear_q.weight,
            k_weight=mha.linear_k.weight,
            v_weight=mha.linear_v.weight,
            gate_weight=mha.linear_g.weight,
            output_weight=mha.linear_o.weight,
            bias_weight=self.linear_z.weight,
            heads=mha.no_heads,
            head_dim=mha.c_hidden,
            inf=self.inf,
            transpose_bias=transpose_bias,
            chunk_size=chunk_size,
        )
        if not self.starting:
            output = output.transpose(-2, -3)
        return output

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        transpose_bias: bool = False,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_triton_triangle_kernels: bool = False,
        use_stable_triangle_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        use_protenix_triangle_attention: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, I, J, C_in] input tensor (e.g. the pair representation)
            mask:
                [*, N_token, N_token] Pair mask
            transpose_bias:
                Whether to transpose the pairwise bias, which would otherwise match the
                input tensor. In practice this is used for Triangle Attention from the
                end node, where the input is transposed prior to calling this function.
                The bias would retain the ordering of the original un-transposed
                input tensor.
            chunk_size:
                Inference-time subbatch size. Acts as a minimum if
                self.tune_chunk_size is True
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory efficient kernel.
                Mutually exclusive with use_lma.
            use_cueq_triangle_kernels:
                Whether to use cuEquivariance triangle multiplicative
                update kernel and attention kernel. When both this and
                use_deepspeed_evo_attention are True, the cuEquivariance
                kernel is only used for triangle attention
            use_lma:
                Whether to use low-memory attention during inference.
                Mutually exclusive with use_deepspeed_evo_attention.
            inplace_safe:
                Whether inplace operations can be performed
        Returns:
            [*, I, J, C_in] output tensor
        """

        if use_protenix_triangle_attention:
            if chunk_size is not None or inplace_safe:
                raise ValueError(
                    "Protenix Triangle Attention requires chunk_size=None and "
                    "inplace_safe=False"
                )
            if any(
                (
                    use_stable_triangle_attention,
                    use_deepspeed_evo_attention,
                    use_triton_triangle_kernels,
                    use_lma,
                )
            ):
                raise ValueError(
                    "Protenix Triangle Attention is mutually exclusive with "
                    "other attention backends"
                )
            use_cueq_triangle_kernels = False

        allow_fp32_inference = False
        if use_stable_triangle_attention:
            from openfold3.core.utils.distributed_contract import (
                validate_turbopairformer_attention_policy,
            )

            # Validate before the fused route can return without visiting the
            # primitive or facade, both of which enforce this same policy.
            validate_turbopairformer_attention_policy()
            # Preserve the original input's eligibility through LayerNorm:
            # no_grad projections erase an input's requires_grad bit.
            allow_fp32_inference = is_inference_without_checkpoint(self, x, mask)
            if chunk_size is not None and not allow_fp32_inference:
                raise ValueError(
                    "stable Triangle Attention chunking is inference-only; "
                    "use eval() with gradients disabled outside checkpointing"
                )
            if self.c_hidden not in (16, 32) or self.no_heads < 1:
                raise ValueError(
                    "stable Triangle Attention production requires c_hidden in "
                    "(16, 32) and no_heads >= 1"
                )
            if any(
                (
                    use_deepspeed_evo_attention,
                    use_triton_triangle_kernels,
                    use_lma,
                )
            ):
                raise ValueError(
                    "stable Triangle Attention is mutually exclusive with all "
                    "other attention backends"
                )
            # use_cueq_triangle_kernels composes: it keeps governing the
            # non-attention triangle kernels (multiplicative update), while
            # stable Triangle Attention owns the attention site.
            use_cueq_triangle_kernels = False
            fused = self._fused_fp32_inference(x, mask, transpose_bias, chunk_size)
            if fused is not None:
                return fused

        if mask is None:
            # [*, I, J]
            mask = x.new_ones(
                x.shape[:-1],
            )

        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        # [*, I, J, C_in]
        x = self.layer_norm(x)

        # [*, I, 1, 1, J]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        # [*, H, I, J]
        if transpose_bias:
            triangle_bias = permute_final_dims(self.linear_z(x), (2, 1, 0))
        else:
            triangle_bias = permute_final_dims(self.linear_z(x), (2, 0, 1))

        # [*, 1, H, I, J]
        triangle_bias = triangle_bias.unsqueeze(-4)

        biases = [mask_bias, triangle_bias]

        if chunk_size is not None:
            x = self._chunk(
                x,
                biases,
                chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                use_triton_triangle_kernels=use_triton_triangle_kernels,
                use_stable_triangle_attention=use_stable_triangle_attention,
                inplace_safe=inplace_safe,
                _allow_fp32_inference=allow_fp32_inference,
            )
        else:
            x = self.mha(
                q_x=x,
                kv_x=x,
                biases=biases,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                use_triton_triangle_kernels=use_triton_triangle_kernels,
                use_stable_triangle_attention=use_stable_triangle_attention,
                use_protenix_triangle_attention=use_protenix_triangle_attention,
                _allow_fp32_inference=allow_fp32_inference,
            )

        if not self.starting:
            x = x.transpose(-2, -3)

        return x


# Implements AF2 Algorithm 13 / AF3 Algorithm 14
TriangleAttentionStartingNode = TriangleAttention


class TriangleAttentionEndingNode(TriangleAttention):
    """
    Implements AF2 Algorithm 14 / AF3 Algorithm 15.
    """

    __init__ = partialmethod(TriangleAttention.__init__, starting=False)
