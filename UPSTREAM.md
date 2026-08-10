# Upstream provenance

- Upstream repository: <https://github.com/aqlaboratory/openfold-3>
- Upstream base commit: `c615a7f8d0e7bb04095a35ec6dc7b30192ac19d3`
- Source development repository: <https://github.com/RyanAIResearch/hammer_inference_engine>
- Source production-candidate commit: `6db02ff4b31e87bfced98331b22613e96e8ced37`
- Curated repository: <https://github.com/RyanAIResearch/openfold3_triangle_kernel>
- License: Apache-2.0

The complete AQLab source tree is retained so that the OpenFold3 model,
training recipe, configuration system, and data pipeline remain runnable. The
RyanAIResearch delta is limited to the StableFold Triangle Attention
runtime, its OpenFold3 routing/configuration integration, package data, and
focused release tests.

The historical kernel experiments and obsolete PyTorch-aligned variants remain
in the development repository and are intentionally excluded here.
