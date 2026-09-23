# TurboPairFormer

TurboPairFormer provides high-performance, autograd-supported GPU kernels for **Triangle Attention** and **Triangle Multiplicative Update** (TriMul), engineered specifically for training large-scale protein folding models.

Developed collaboratively by researchers from **Stevens Institute of Technology**, **Lambda**, and **OpenFold**.

---

## Requirements

- **GPU**: NVIDIA Hopper architecture (`sm_90a`, H100 validated; H200 experimental)
- **Environment**: Linux x86-64 (glibc ≥ 2.28), Python 3.14, PyTorch 2.10.0 with CUDA Runtime 12.8
- **No CUDA Toolkit Required**: Pre-compiled AOT wheels include the kernel binaries.
- **Binary Kernel Distribution**: Wheels include compiled CUDA kernels and Python interfaces; CUDA/C++ source files are not included.

---

## Installation

```bash
conda create -n turbopairformer python=3.14 pip -y
conda activate turbopairformer

export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r

pip install turbopairformer
```

Install this binary-kernel release explicitly:

```bash
pip install turbopairformer==0.1.2
```

## Offline CLI Documentation

TurboPairFormer includes built-in offline API documentation. It requires no PyTorch import, GPU, compiler, or network connection:

```bash
# Explore the two operator contracts
turbopairformer --help
python -m turbopairformer --help
turbopairformer api turbo_attention
turbopairformer api turbo_trimul

# View both operator references
turbopairformer api --all
```

## Quick Start

### Triangle Attention

`turbo_attention` accelerates the triangle attention mechanism with native autograd support for Q, K, V, and pair bias.

```python
import torch
from turbopairformer import turbo_attention

# Inputs layout: [*B, O, H, N, D]
# Constraints: 1 <= D <= 64; Q is unscaled (kernel applies 1/sqrt(D) internally)
options = dict(device="cuda", dtype=torch.bfloat16, requires_grad=True)
q, k, v = [torch.randn(1, 128, 4, 128, 32, **options) for _ in range(3)]

# pair_bias is required: [*B, 1, H, N, N] (pass zeros if your architecture has no bias)
pair_bias = torch.randn(1, 1, 4, 128, 128, **options)

output = turbo_attention(q, k, v, pair_bias=pair_bias)
output.float().square().mean().backward()

# Returned layout swaps sequence and head dimensions: [*B, O, N, H, D]
print(output.shape)  # torch.Size([1, 128, 128, 4, 32])
```

**Masking Note:** `res_mask` must be an additive floating mask (`[*B, O, 1, 1, N]`), where `0.0` keeps keys and `-inf` masks them out. Do not pass boolean masks.

### Triangle Multiplicative Update (TriMul)

`turbo_trimul` uses optimized Hopper kernels for normalization, projection, and triangle multiplicative updates.

```python
import torch
from turbopairformer import turbo_trimul

# Input z: [*B, N, N, C] where N > 100 and C in {32, 64, 96, 128}
z = torch.randn(
    1, 128, 128, 128,
    device="cuda", dtype=torch.bfloat16, requires_grad=True,
)

# direction accepts "outgoing" (default) or "incoming"
output = turbo_trimul(z, direction="outgoing")
output.float().square().mean().backward()

print(output.shape)  # torch.Size([1, 128, 128, 128])
```

**Training vs. Sanity Check:** When called without weights as shown above, projection and gate matrices are sampled per call; normalization weights and biases default to ones and zeros. These temporary parameters do not require gradients. For actual training, explicitly supply your persistent module weights (`norm_in_weight`, `p_in_weight`, etc.).

## Project Team

- **Yide Ran** ¹
- **Chelsea Lowman** ²
- **Jan Domanski** ³
- **David Hartmann** ²
- **Jenke Scheen** ³
- **Jennifer Wei** ³
- **Chuan Li** ²
- **Jianwen Xie** ² ⁺
- **Zhaozhuo Xu** ¹ ⁺

### Affiliations

- ¹ Stevens Institute of Technology
- ² Lambda
- ³ OpenFold
- ⁺ Equal advising

## License

Apache-2.0
