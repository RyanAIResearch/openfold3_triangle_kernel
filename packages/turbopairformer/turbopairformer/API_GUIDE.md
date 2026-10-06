# TurboPairFormer API usage

## attention - Triangle Attention

Single-reconstruction backward is mandatory in the current source. The effective
`TURBOPAIRFORMER_SR_BACKWARD` value must be unset or `1`; `0` raises.

```python
from turbopairformer import turbo_attention

output = turbo_attention(q, k, v, res_mask=None, pair_bias=pair_bias)
```

- `q/k/v`: BF16 on Hopper CUDA, shape `[*B, O, H, N, D]`; Q is unscaled, `1 <= D <= 64`.
  `*B`: batch dimensions (at least one); `H`: heads; `N`: sequence; `D`: head size.
  `O`: outer dimension the `pair_bias` is shared across.
- `pair_bias`: required BF16/FP32 `[*B, 1, H, N, N]`; batch dimensions may broadcast.
- `res_mask`: optional additive FP32/BF16/FP16 `[*B, O, 1, 1, N]`; batch dimensions may broadcast.
  `0` keeps a key, `-inf` masks it; `None` keeps all keys.
- Returns BF16 `[*B, O, N, H, D]` (N/H axes exchanged), with Q/K/V/bias gradients.

Unsupported tensor inputs raise `ValueError`; precheck with `is_triangle_attention_supported(...)`.

## trimul - Triangle Multiplicative Update

```python
from turbopairformer import turbo_trimul

output = turbo_trimul(z, direction="outgoing", mask=None, **weights)
output = turbo_trimul(module, z, mask)  # OpenFold3-style module carries the weights
```

- `z`: BF16 on Hopper CUDA, shape `[*B, N, N, C]`; `N > 100`, `C` in `{32, 64, 96, 128}`; batch dimensions optional.
- `direction`: `"outgoing"` or `"incoming"`.
- `mask`: optional bool/BF16/FP32 pair mask broadcastable to `[*B, N, N]` on the same device.
  `1` keeps a pair, `0` masks it; `None` keeps all pairs.
- Returns the same shape/dtype as `z`, with first-order input/parameter gradients.

`weights`: eight contiguous FP32 parameters on `z.device`:

- `norm_in_weight`, `norm_in_bias`, `norm_out_weight`, `norm_out_bias`: `[C]` each.
- `p_in_weight`, `g_in_weight`: `[2*C, C]` each (branch a, then b).
- `p_out_weight`, `g_out_weight`: `[C, C]` each.
Training uses persistent weights; omitted weights are per-call demonstration defaults.
`eps` must be `1e-5`. Neither operator differentiates masks.

Unsupported tensor inputs raise `ValueError`; precheck module calls with `check_trimul_support(module, z, mask)`.
