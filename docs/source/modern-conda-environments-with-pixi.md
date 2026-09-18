# Modern OpenFold conda environments with pixi

Pixi is a modern tool that allows for much faster environment building over conda, Pixes also comes with .lock environment files out of the box. 

Key terms when working with pixi
- an "environment", it's a group of "features", an example environment is `openfold3-cuda13-pypi`
- a "feature" is a group of packages/dependencies (either from conda or pypi)
- each environment is then composed from one or more features 

Here is the upstream overview diagram showing the environment and feature
composition. In this fork, the editable CUDA 12 development environment shown in
the diagram is named `openfold3-cuda12-dev`; the shorter `openfold3-cuda12` name
is reserved for the pinned, non-editable TurboPairFormer H100 runtime
described in {doc}`Installation <Installation>`.

```{figure} ../imgs/pixi-environments-and-features.png
:class: only-light
:alt: Pixi for OpenFold – environments and features
```
