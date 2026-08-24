# H100 direct-delta comparison

Date: 2026-08-23 UTC

GPU: NVIDIA H100 80GB HBM3

Kernel commit: `4be8da30152591ff94c8b95cbe84dda0c3ebb37a`

Evidence root: `/home/ubuntu/xzz/of3_perf_runs/iclr27_h100/direct_delta_4be8da301_20260823T232759Z`

## Design

The gate evaluates four delta contracts directly on 26 hashed, real pretrained
activation captures: classic output delta, BF16 residual-corrected shadow,
uncompressed FP32 output shadow, and an FP64-perfect delta cast to FP32. The
captures comprise eight N640/D16, six N384/D16, six N640/D32 skip20, four
N640/D32 skip60, and two N640 APB O1/H16/D24 cases. The same SR backward then
propagates each delta to dQ/dK/dV/dBias for comparison with the perfect arm.

## Result and claim boundary

- FP32 shadow is not bitwise equal to the perfect delta: 0/26 exact cases.
- Its direct-delta relative L2 is 4.21574e-7 mean and 7.84388e-7 maximum.
- Its downstream dQ error-floor ratio to perfect is 1.001361 mean and 1.032273
  maximum.
- BF16 shadow recovers 83.3438% mean and 88.1451% median of the
  classic-to-perfect dQ gap.

The defensible statement is therefore: **FP32 shadow reaches the same observed
downstream dQ error floor as the perfect-delta oracle across these captures.**
It must not be described as equal or bitwise equal to perfect delta.

| Cohort | n | Classic delta rel-L2 | BF16 delta rel-L2 | FP32 delta rel-L2 | BF16 dQ gap recovery | FP32/perfect dQ floor |
|---|---:|---:|---:|---:|---:|---:|
| N384 D16 | 6 | 1.371e-3 | 4.494e-4 | 4.316e-7 | 74.1% | 1.000083x |
| APB D24 | 2 | 2.613e-4 | 7.407e-5 | 7.806e-8 | 76.4% | 1.017363x |
| N640 D16 | 8 | 1.485e-3 | 4.262e-4 | 5.847e-7 | 89.9% | 1.000001x |
| N640 D32 skip20 | 6 | 1.367e-3 | 4.260e-4 | 3.395e-7 | 81.0% | 1.000032x |
| N640 D32 skip60 | 4 | 1.291e-3 | 3.649e-4 | 3.752e-7 | 91.0% | 0.999992x |

Canonical summaries are `analysis/direct_delta_cohorts.md` and the associated
JSON/JSONL files. The evidence manifest contains 20 files and verifies with
`sha256sum -c`; its own SHA256 is
`0fe5c67199c62d5282798171bc3e3563f5a76f29af8966c35b16e807b12dc361`.
