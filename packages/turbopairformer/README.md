# TurboPairFormer

TurboPairFormer provides optimized GPU kernels for **Triangle Attention** and **Triangle Multiplicative Update** (TriMul), with PyTorch autograd support for protein folding model training.

The project brings together researchers from Stevens Institute of Technology, Lambda, and OpenFold.

---

## Installation

```bash
conda create -n turbopairformer python=3.14 pip -y
conda activate turbopairformer

export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r

pip install turbopairformer
```
