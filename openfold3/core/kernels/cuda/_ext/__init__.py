"""Private AOT CUDA extension modules for StableFold kernels.

The binary modules are installed by the Hopper ``sm_90a`` wheel.  They are
kept in a private package so their CPython initializer names cannot collide
with the historical top-level modules created by ``cpp_extension.load``.
"""
