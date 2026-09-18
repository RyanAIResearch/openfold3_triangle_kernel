# Upstream provenance

- Upstream repository: <https://github.com/aqlaboratory/openfold-3>
- Latest synchronized upstream commit (2026-09-18): `7de748b7bc93adb5af0a4032d5e9f208b6e1325f`
- Previous upstream base commit: `e47adbf95448956574603c9e8eb703cbb0c3bb0b`
- Previous upstream synchronization merge: `60867fb6f0d1949cfe0ade01fb8e458cee1aeab8`
- Source development repository: <https://github.com/RyanAIResearch/hammer_inference_engine>
- Source production-candidate commit: `6db02ff4b31e87bfced98331b22613e96e8ced37`
- Curated repository: <https://github.com/RyanAIResearch/openfold3_triangle_kernel>
- License: Apache-2.0

The complete AQLab source tree is retained so that the OpenFold3 model,
training recipe, configuration system, and data pipeline remain runnable. The
RyanAIResearch delta is limited to the TurboPairFormer (formerly StableFold)
Triangle Attention and Triangle Multiplicative Update runtimes, their routing/configuration
integration, AOT package data, a kernel-only DDP identity preflight, and focused
release tests. AQLab's DDP strategy, gradient reduction, EMA, loss, checkpoint,
logging, optimizer, data pipeline, and training recipe remain unchanged.

The historical kernel experiments and obsolete PyTorch-aligned variants remain
in the development repository and are intentionally excluded here.

The 2026-09-18 synchronization retains the fork's kernel routes, CUDA source,
and pinned production wheel. Upstream inference/data/metrics fixes and the
Pixi-based CI/Docker changes are merged into the host source. The current
production wheel remains the artifact built from `87ebe22da`; this source
merge does not rebuild it or extend its hardware/training qualification.
