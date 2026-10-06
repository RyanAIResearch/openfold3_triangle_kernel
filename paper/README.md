# TurboPairFormer methodology and evidence

TurboPairFormer implements Triangle Attention and Triangle Multiplicative
Update for OpenFold3. The current source tree uses the new namespace throughout;
new binaries and their installed execution paths require new validation.

## Method outline

Triangle Attention backward has distinct reduction owners for query, key/value
and broadcast pair-bias gradients. The single-reconstruction design shares an
attention tile's reconstructed probabilities and score derivatives among those
gradient producers. Ordered partial reductions make arithmetic order explicit,
while workspace planning controls how partials are stored and scheduled.

The optional forward rounding residual supplies additional information to the
backward row-scalar calculation without changing the public BF16 forward
output. It addresses that numerical error channel; it is not a claim that all
BF16 error disappears. Triangle Multiplicative Update has its own execution and
numerical contracts and must be validated independently.

## Evidence boundaries

Operator timing, whole training-step timing, numerical error, repeatability,
peak memory and model convergence answer different questions. A release report
must identify its source commit, wheel and extension hashes, runtime ABI, GPU,
input/gradient scope, timing method and statistical unit. Determinism claims
must specify the exact inputs, arithmetic schedule, binary and hardware tested.

Do not use historical timing or correctness receipts as certification of a
renamed binary. Current packaging receipts cover only the checks they actually
record; full training and multi-GPU qualification remain separate gates.

## Historical material

The earlier manuscript draft, outline, experiment manifest, raw receipt archive
and result figures are preserved unmodified in the
[archived source snapshot](https://github.com/RyanAIResearch/openfold3_triangle_kernel/tree/a522470b500d040e5649e1deed1a7a9581438c62). The dated protocols and limitations there
remain part of those original experiments. The current tree retains this
method outline instead of rewriting their identities or presenting old numbers
as new release results.
