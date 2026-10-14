#!/usr/bin/env python3
"""Minimal full-F+B smoke for an installed, compiler-free AOT wheel."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch

MODULES = (
    "openfold3.core.kernels.turbopairformer._ext._turbopairformer_forward_sm90a",
    "openfold3.core.kernels.turbopairformer._ext._turbopairformer_delta_sm90a",
    "openfold3.core.kernels.turbopairformer._ext._turbopairformer_dkdv_sm90a",
    "openfold3.core.kernels.turbopairformer._ext._turbopairformer_dq_dbias_sm90a",
    "openfold3.core.kernels.turbopairformer._ext._turbopairformer_trimul_sm90a",
)

FORWARD_SCHEMA = (
    "openfold3_turbopairformer::_trimul_fwd_v1(Tensor z_bf, Tensor mask_f, "
    "Tensor w_ap, Tensor w_ag, Tensor w_bp, Tensor w_bg, Tensor w_g, "
    "Tensor w_z, Tensor g_in, Tensor b_in, Tensor g_out, Tensor b_out, "
    "bool outgoing, bool save_state) -> (Tensor, Tensor, Tensor, Tensor, "
    "Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
)
BACKWARD_SCHEMA = (
    "openfold3_turbopairformer::_trimul_bwd_v1(Tensor grad_out, Tensor z_bf, "
    "Tensor mask_f, Tensor z_ln, Tensor gate, Tensor a, Tensor b, Tensor x, "
    "Tensor xlnT, Tensor mean_in, Tensor rstd_in, Tensor mean_out, "
    "Tensor rstd_out, Tensor w_ap, Tensor w_ag, Tensor w_bp, Tensor w_bg, "
    "Tensor w_g, Tensor w_z, Tensor g_in, Tensor g_out, Tensor out, "
    "bool outgoing) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
)


def raw_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu().view(torch.uint8).numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def assert_no_jit_cache(stage: str) -> None:
    cache_value = os.environ.get("TORCH_EXTENSIONS_DIR")
    if cache_value is None:
        raise RuntimeError("clean-install smoke requires a forbidden JIT cache path")
    if Path(cache_value).exists():
        raise RuntimeError(f"{stage}: production path touched the forbidden JIT cache")


def main() -> None:
    if shutil.which("nvcc") is not None:
        raise RuntimeError("clean-install smoke requires nvcc to be absent from PATH")
    if os.environ.get("OPENFOLD3_TRIANGLE_ALLOW_JIT") is not None:
        raise RuntimeError("clean-install smoke forbids the development JIT switch")
    assert_no_jit_cache("startup")

    binaries = [importlib.import_module(name) for name in MODULES]
    binary_paths = [Path(module.__file__).resolve() for module in binaries]
    if any(path.suffix != ".so" for path in binary_paths):
        raise RuntimeError(f"expected five AOT shared objects, got {binary_paths}")
    assert_no_jit_cache("AOT binary import")

    from openfold3.core.kernels.turbopairformer import turbopairformer_trimul as stable

    if stable._ext is not None:
        raise RuntimeError("custom-op registration eagerly loaded the TriMul facade")
    forward_schema = str(torch.ops.openfold3_turbopairformer._trimul_fwd_v1.default._schema)
    backward_schema = str(torch.ops.openfold3_turbopairformer._trimul_bwd_v1.default._schema)
    if forward_schema != FORWARD_SCHEMA or backward_schema != BACKWARD_SCHEMA:
        raise RuntimeError(
            "versioned TurboPairFormer TriMul schemas differ from the release contract"
        )
    assert_no_jit_cache("custom-op import")

    device = torch.device("cuda:0")
    channels = 32

    class InstalledTriMul(torch.nn.Module):
        """Exact supported module contract used by clean-install smoke gates."""

        def __init__(self, outgoing: bool):
            super().__init__()
            self.c_z = channels
            self.c_hidden = channels
            self._outgoing = outgoing

            def linear() -> torch.nn.Linear:
                return torch.nn.Linear(
                    channels,
                    channels,
                    bias=False,
                    device=device,
                    dtype=torch.float32,
                )

            self.linear_a_p = linear()
            self.linear_a_g = linear()
            self.linear_b_p = linear()
            self.linear_b_g = linear()
            self.linear_g = linear()
            self.linear_z = linear()
            self.layer_norm_in = torch.nn.LayerNorm(
                channels,
                eps=1e-5,
                device=device,
                dtype=torch.float32,
            )
            self.layer_norm_out = torch.nn.LayerNorm(
                channels,
                eps=1e-5,
                device=device,
                dtype=torch.float32,
            )

        def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return stable._run_supported_stable_trimul(self, z, mask)

    fake_sequence = 17
    fake_z = torch.empty(
        (1, fake_sequence, fake_sequence, channels),
        device="meta",
        dtype=torch.bfloat16,
    )
    fake_mask = torch.empty(
        (1, fake_sequence, fake_sequence),
        device="meta",
        dtype=torch.float32,
    )
    fake_weights = tuple(
        torch.empty((channels, channels), device="meta") for _ in range(6)
    )
    fake_norms = tuple(torch.empty((channels,), device="meta") for _ in range(4))
    fake_outputs = stable._trimul_fwd_v1(
        fake_z, fake_mask, *fake_weights, *fake_norms, True, True
    )
    fake_backward_outputs = stable._trimul_bwd_v1(
        torch.empty_like(fake_z),
        fake_z,
        fake_mask,
        *fake_outputs[1:],
        *fake_weights,
        fake_norms[0],
        fake_norms[2],
        fake_outputs[0],
        True,
    )
    fake_rows = fake_sequence * fake_sequence
    expected_forward_shapes = (
        (1, fake_sequence, fake_sequence, channels),
        (fake_rows, channels),
        (fake_rows, channels),
        (channels, fake_rows),
        (channels, fake_rows),
        (channels, fake_sequence, fake_sequence),
        (channels, fake_rows),
        (fake_rows,),
        (fake_rows,),
        (fake_rows,),
        (fake_rows,),
    )
    expected_backward_shapes = (
        (1, fake_sequence, fake_sequence, channels),
        (5 * channels, channels),
        (channels, channels),
        (channels,),
        (channels,),
        (channels,),
        (channels,),
    )
    expected_forward_dtypes = (torch.bfloat16,) * 7 + (torch.float32,) * 4
    expected_backward_dtypes = (torch.bfloat16,) + (torch.float32,) * 6
    if tuple(tuple(value.shape) for value in fake_outputs) != expected_forward_shapes:
        raise RuntimeError("versioned forward fake metadata differs from its schema")
    if tuple(value.dtype for value in fake_outputs) != expected_forward_dtypes:
        raise RuntimeError("versioned forward fake dtypes differ from its schema")
    if (
        tuple(tuple(value.shape) for value in fake_backward_outputs)
        != expected_backward_shapes
    ):
        raise RuntimeError("versioned backward fake metadata differs from its schema")
    if (
        tuple(value.dtype for value in fake_backward_outputs)
        != expected_backward_dtypes
    ):
        raise RuntimeError("versioned backward fake dtypes differ from its schema")
    if stable._ext is not None:
        raise RuntimeError("custom-op fake dispatch loaded the TriMul AOT facade")
    assert_no_jit_cache("custom-op fake")

    def run_dynamic_export() -> dict[str, object]:
        torch.manual_seed(20260831)
        module = InstalledTriMul(True).requires_grad_(False).eval()
        sample_z = torch.randn(
            1,
            17,
            17,
            channels,
            device=device,
            dtype=torch.bfloat16,
        )
        sample_mask = torch.ones(1, 17, 17, device=device, dtype=torch.float32)
        sequence_dim = torch.export.Dim("sequence", min=1, max=256)
        exported = torch.export.export(
            module,
            (sample_z, sample_mask),
            dynamic_shapes={
                "z": {1: sequence_dim, 2: sequence_dim},
                "mask": {1: sequence_dim, 2: sequence_dim},
            },
        )
        custom_targets = [
            str(node.target)
            for node in exported.graph.nodes
            if node.op == "call_function" and "openfold3_turbopairformer" in str(node.target)
        ]
        expected_target = "openfold3_turbopairformer._trimul_fwd_v1.default"
        if custom_targets != [expected_target]:
            raise RuntimeError(
                f"dynamic export has unexpected TurboPairFormer targets: {custom_targets}"
            )
        with tempfile.TemporaryDirectory(
            prefix="openfold3-trimul-export-"
        ) as temporary:
            artifact = Path(temporary) / "trimul_dynamic.pt2"
            torch.export.save(exported, artifact)
            loaded = torch.export.load(artifact).module()
            if stable._ext is not None:
                raise RuntimeError("dynamic export/save/load eagerly loaded TriMul AOT")
            assert_no_jit_cache("dynamic export registration")
            dynamic_z = torch.randn(
                1,
                33,
                33,
                channels,
                device=device,
                dtype=torch.bfloat16,
            )
            dynamic_mask = torch.ones(1, 33, 33, device=device, dtype=torch.float32)
            with torch.no_grad():
                expected = module(dynamic_z, dynamic_mask)
                actual = loaded(dynamic_z, dynamic_mask)
        if not torch.equal(actual, expected):
            raise RuntimeError("dynamic export/save/load differs from eager inference")
        assert_no_jit_cache("dynamic export execution")
        return {
            "target": expected_target,
            "sample_sequence": 17,
            "loaded_sequence": 33,
            "output": raw_sha256(actual),
        }

    dynamic_export_receipt = run_dynamic_export()

    from openfold3.core.kernels.turbopairformer.turbopairformer_triangle_attention import (  # noqa: E501
        invocation_count,
        turbopairformer_triangle_attention,
    )

    # Exact 64-key tiles deliberately exercise the non-contiguous fused QKV
    # pack. The old N=65 smoke padded Q/K/V first and missed this production
    # fast path entirely.
    prefix, sequence, heads, head_dim = 1, 64, 4, 32

    def run_arm(arm: str) -> dict[str, str]:
        if arm == "explicit_sr_shadow":
            os.environ["OPENFOLD3_SR_BACKWARD"] = "1"
            os.environ["OPENFOLD3_SR_SHADOW"] = "1"
        else:
            os.environ.pop("OPENFOLD3_SR_BACKWARD", None)
            os.environ.pop("OPENFOLD3_SR_SHADOW", None)
        torch.manual_seed(20260809)

        def projection_view() -> torch.Tensor:
            base = torch.randn(
                prefix,
                sequence,
                sequence,
                heads,
                head_dim,
                device=device,
                dtype=torch.bfloat16,
            )
            return base.transpose(-2, -3).detach().requires_grad_(True)

        q, k, v = projection_view(), projection_view(), projection_view()
        q_contiguous, k_contiguous, v_contiguous = (
            tensor.contiguous().detach().requires_grad_(True) for tensor in (q, k, v)
        )
        mask = torch.zeros(
            prefix,
            sequence,
            1,
            1,
            sequence,
            device=device,
            dtype=torch.bfloat16,
        )
        bias = torch.randn(
            prefix,
            1,
            heads,
            sequence,
            sequence,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        bias_contiguous = bias.detach().clone().requires_grad_(True)
        before = invocation_count()
        output = turbopairformer_triangle_attention(q, k, v, mask, bias)
        output_contiguous = turbopairformer_triangle_attention(
            q_contiguous,
            k_contiguous,
            v_contiguous,
            mask,
            bias_contiguous,
        )
        d_output = torch.randn_like(output)
        gradients = torch.autograd.grad(output, (q, k, v, bias), d_output)
        gradients_contiguous = torch.autograd.grad(
            output_contiguous,
            (q_contiguous, k_contiguous, v_contiguous, bias_contiguous),
            d_output,
        )
        torch.cuda.synchronize()
        named = {
            "output": output,
            "dQ": gradients[0],
            "dK": gradients[1],
            "dV": gradients[2],
            "dBias": gradients[3],
        }
        if invocation_count() - before != 2:
            raise RuntimeError(
                f"{arm}: public facade route counter did not advance twice"
            )
        if not all(torch.isfinite(value).all().item() for value in named.values()):
            raise RuntimeError(f"{arm}: AOT full F+B produced a non-finite result")
        reference = {
            "output": output_contiguous,
            "dQ": gradients_contiguous[0],
            "dK": gradients_contiguous[1],
            "dV": gradients_contiguous[2],
            "dBias": gradients_contiguous[3],
        }
        mismatched = [
            name
            for name, value in named.items()
            if not torch.equal(value, reference[name])
        ]
        if mismatched:
            raise RuntimeError(
                f"{arm}: non-contiguous fused pack differs from the contiguous "
                "path: " + ", ".join(mismatched)
            )
        return {name: raw_sha256(value) for name, value in named.items()}

    arm_receipts = {}
    # Both arms use SR + shadow: one selects the defaults, the other sets the
    # flags explicitly. The current facade has no public two-pass backward.
    for arm in ("default_sr_shadow", "explicit_sr_shadow"):
        arm_receipts[arm] = run_arm(arm)

    def run_trimul(outgoing: bool) -> dict[str, object]:
        torch.manual_seed(20260830 + int(outgoing))
        sequence = 101
        module = InstalledTriMul(outgoing)
        named_parameters = (
            ("w_ap", module.linear_a_p.weight),
            ("w_ag", module.linear_a_g.weight),
            ("w_bp", module.linear_b_p.weight),
            ("w_bg", module.linear_b_g.weight),
            ("w_g", module.linear_g.weight),
            ("w_z", module.linear_z.weight),
            ("g_in", module.layer_norm_in.weight),
            ("b_in", module.layer_norm_in.bias),
            ("g_out", module.layer_norm_out.weight),
            ("b_out", module.layer_norm_out.bias),
        )
        z_value = torch.randn(
            1,
            sequence,
            sequence,
            channels,
            device=device,
            dtype=torch.bfloat16,
        )
        mask = torch.ones(1, sequence, sequence, device=device, dtype=torch.float32)
        d_output = torch.randn_like(z_value)

        def execute(runner) -> dict[str, torch.Tensor]:
            z = z_value.detach().clone().requires_grad_(True)
            output = runner(z)
            gradients = torch.autograd.grad(
                output,
                (z, *(parameter for _, parameter in named_parameters)),
                d_output,
            )
            named = {"output": output, "dZ": gradients[0]}
            named.update(
                {
                    f"d{name}": gradient
                    for (name, _), gradient in zip(
                        named_parameters, gradients[1:], strict=True
                    )
                }
            )
            return named

        eager = execute(lambda z: stable.run_stable_trimul(module, z, mask))
        compiled_module = torch.compile(
            module,
            backend="aot_eager",
            fullgraph=True,
        )
        compiled = execute(lambda z: compiled_module(z, mask))
        torch.cuda.synchronize()
        if not all(
            torch.isfinite(value).all().item()
            for receipt in (eager, compiled)
            for value in receipt.values()
        ):
            direction = "outgoing" if outgoing else "incoming"
            raise RuntimeError(
                f"trimul {direction}: AOT full F+B produced a non-finite result"
            )
        mismatched = [
            name
            for name, value in eager.items()
            if not torch.equal(value, compiled[name])
        ]
        if mismatched:
            direction = "outgoing" if outgoing else "incoming"
            raise RuntimeError(
                f"trimul {direction}: aot_eager fullgraph F+B differs from eager: "
                + ", ".join(mismatched)
            )
        assert_no_jit_cache("TriMul eager and aot_eager")
        return {
            "sequence": sequence,
            "eager": {name: raw_sha256(value) for name, value in eager.items()},
            "aot_eager": {name: raw_sha256(value) for name, value in compiled.items()},
        }

    def run_short_sequence_stock() -> dict[str, object]:
        from openfold3.core.kernels.turbopairformer.turbopairformer_trimul_support import (  # noqa: E501
            check_stable_trimul_support,
        )
        from openfold3.core.model.layers import (
            triangular_multiplicative_update as model_module,
        )
        from openfold3.core.model.layers.triangular_multiplicative_update import (
            TriangleMultiplicationOutgoing,
        )

        torch.manual_seed(20260901)
        sequence = 17
        module = TriangleMultiplicationOutgoing(channels, channels).to(device).train()
        with torch.no_grad():
            for name, parameter in module.named_parameters():
                if "layer_norm" in name and name.endswith("weight"):
                    parameter.normal_(mean=1.0, std=0.05)
                else:
                    parameter.normal_(mean=0.0, std=0.02)
        named_parameters = tuple(module.named_parameters())
        z_value = torch.randn(
            1,
            sequence,
            sequence,
            channels,
            device=device,
            dtype=torch.bfloat16,
        )
        mask = torch.ones(1, sequence, sequence, device=device, dtype=torch.float32)
        d_output = torch.randn_like(z_value)

        def execute(use_stable: bool) -> dict[str, torch.Tensor]:
            z = z_value.detach().clone().requires_grad_(True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = module(
                    z,
                    mask=mask,
                    inplace_safe=False,
                    use_cueq_triangle_kernels=use_stable,
                    use_triton_triangle_kernels=use_stable,
                    use_stable_triangle_multiplicative=use_stable,
                )
            gradients = torch.autograd.grad(
                output,
                (z, *(parameter for _, parameter in named_parameters)),
                d_output,
            )
            named = {"output": output, "d_z": gradients[0]}
            named.update(
                {
                    f"d_{name}": gradient
                    for (name, _), gradient in zip(
                        named_parameters, gradients[1:], strict=True
                    )
                }
            )
            return named

        stock = execute(False)
        support = check_stable_trimul_support(module, z_value, mask)
        original_fallback_logged = set(model_module._TURBOPAIRFORMER_TRIMUL_FALLBACK_LOGGED)
        if support or support.code != "sequence_stock_fallback":
            raise RuntimeError(
                "N=17 must be metadata-unsupported by the TurboPairFormer AOT route"
            )

        counts = {"runner": 0, "get_ext": 0, "cueq": 0}
        original_runner = stable._run_supported_stable_trimul
        original_get_ext = stable._get_ext
        original_cueq = model_module._cueq_triangle_mult
        original_triton_available = model_module.TRITON_AVAILABLE

        def forbidden(name: str):
            def wrapped(*args, **kwargs):
                counts[name] += 1
                raise RuntimeError(f"N=17 stock selector entered {name}")

            return wrapped

        stable._run_supported_stable_trimul = forbidden("runner")
        stable._get_ext = forbidden("get_ext")
        model_module._cueq_triangle_mult = forbidden("cueq")
        model_module.TRITON_AVAILABLE = False
        try:
            fallback = execute(True)
        finally:
            stable._run_supported_stable_trimul = original_runner
            stable._get_ext = original_get_ext
            model_module._cueq_triangle_mult = original_cueq
            model_module.TRITON_AVAILABLE = original_triton_available
            model_module._TURBOPAIRFORMER_TRIMUL_FALLBACK_LOGGED.clear()
            model_module._TURBOPAIRFORMER_TRIMUL_FALLBACK_LOGGED.update(
                original_fallback_logged
            )

        if any(counts.values()):
            raise RuntimeError(f"N=17 did not remain on stock PyTorch: {counts}")
        mismatched = [
            name
            for name, value in fallback.items()
            if not torch.equal(value, stock[name])
        ]
        if mismatched:
            raise RuntimeError(
                "N=17 TurboPairFormer selector differs from stock PyTorch: "
                + ", ".join(mismatched)
            )
        assert_no_jit_cache("TriMul N=17 stock selector")
        return {
            "sequence": sequence,
            "support_code": support.code,
            "route_counts": counts,
            "fields": {name: raw_sha256(value) for name, value in fallback.items()},
        }

    trimul_receipts = {
        "outgoing": run_trimul(True),
        "incoming": run_trimul(False),
    }
    short_sequence_stock_receipt = run_short_sequence_stock()

    from openfold3.core.kernels.turbopairformer._aot_runtime import (
        distributed_aot_identity,
    )
    from openfold3.core.utils.distributed_contract import (
        installed_distribution_identity,
    )

    aot_identity = distributed_aot_identity(
        (
            "turbopairformer_forward",
            "turbopairformer_delta",
            "turbopairformer_dkdv",
            "turbopairformer_dq_dbias",
            "turbopairformer_trimul",
        )
    )
    distribution_identity = installed_distribution_identity()

    assert_no_jit_cache("final")
    print(
        json.dumps(
            {
                "verdict": "PASS",
                "device": torch.cuda.get_device_name(0),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "binaries": [str(path) for path in binary_paths],
                "aot_identity": aot_identity,
                "installed_distribution": distribution_identity,
                "arms": arm_receipts,
                "trimul": trimul_receipts,
                "trimul_short_sequence_stock": short_sequence_stock_receipt,
                "trimul_custom_ops": {
                    "forward_schema": forward_schema,
                    "backward_schema": backward_schema,
                    "fake_forward_shapes": [
                        list(value.shape) for value in fake_outputs
                    ],
                    "fake_backward_shapes": [
                        list(value.shape) for value in fake_backward_outputs
                    ],
                    "compile_backend": "aot_eager",
                    "dynamic_export": dynamic_export_receipt,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
