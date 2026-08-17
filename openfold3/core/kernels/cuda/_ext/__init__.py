"""Private AOT CUDA extension modules for StableFold Triangle Attention.

The four binary modules are installed by the H100 ``sm_90a`` wheel.  They are
kept in a private package so their CPython initializer names cannot collide
with the historical top-level modules created by ``cpp_extension.load``.
"""
