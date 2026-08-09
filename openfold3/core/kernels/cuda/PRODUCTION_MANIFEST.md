# Lambda Streaming Triangle Attention — release manifest

本文件描述当前由 `use_stable_triangle_attention=true` 选择的 Lambda
Triangle Attention 路径。OpenFold3 默认配置仍保持官方实现；Lambda 路径必须
显式启用，并且对不支持的输入直接报错，不会静默回退 PyTorch、AQLab Triton
或 NVIDIA cuEquivariance。

## 当前发布状态

**状态：production candidate（尚未 production-ready）。**

当前已经具备：

- package-owned public autograd facade 与 CUDA 源码；
- 四个互不冲突的 AOT CUDA extension module 与 production fail-closed loader；
- online-softmax forward；
- zero-atomics recompute backward；
- 同一输入的逐 bit self-replay；
- CUDA device、dtype、shape、grid 与 stream 的 fail-closed 检查；
- generic runtime sequence/head 路径与非整 tile 尾块保护；
- 将任意正 prefix flatten 为一个 CUDA batch；broadcast dBias 由固定顺序
  CUDA kernel 归约；
- 以 query slab、batch chunk 和 leaf wave 限定临时显存，同时保持同一条
  全局 dBias 归约顺序；
- fully-masked real `-inf` 行的零输出、零梯度契约；
- H100 Compute Sanitizer component smoke。

正式放行仍缺少：

1. release commit 的 H100 AOT wheel 动态构建、二进制核验和无编译器
   clean-install 回执；
2. 完整 Compute Sanitizer release suite；
3. 多 GPU/DDP 正确性与性能证据；
4. 长期训练轨迹、validation quality 与 time-to-quality。

原计划的 1000-step training soak 未在本发布周期内执行。这不构成失败证据，
但也不能替代长期收敛证据；因此当前状态仍是 production candidate。

## 运行时支持范围

- GPU：Hopper `sm_90`（当前发布门禁设备为 Lambda H100 80GB）；
- dtype：Q/K/V 与 pair bias 为 BF16；mask 为 FP32/BF16/FP16；
- head dimension：production 只接受 `D in {16, 32}`；
- sequence/head：设备可行范围内的 runtime `N > 0`、`H > 0`；
- CUDA grid 边界：`ceil(N/32) <= 65535`、`H <= 65535`、flatten 后
  `B*N <= 2^31-1`；实际可用上限还受 `O(N^2)` 输入/输出显存与运行时间限制；
- batch prefix：任意非空、可广播的正 prefix 会 flatten 为一个 CUDA batch，
  不再逐 leaf Python 调度；
- pair bias：若确实需要 broadcast，target prefix rank 必须 `<=16`；若 bias
  prefix 已与 target prefix 完全一致，则没有这一 broadcast bookkeeping 限制。

`N=384/640/768` 是编译期快车道，不是功能白名单。其他长度走
`StaticSequence=0` generic kernel，并使用显式尾块 guard；不会因为长度不同而
回退其他后端。

## 生产调用链

| 文件 | 职责 |
|---|---|
| `openfold3/core/model/primitives/attention.py` | 显式 Lambda 路由；unsupported fail-closed |
| `streaming_triangle_attention.py` | public facade、broadcast/prefix 编排、自定义 autograd |
| `_aot_runtime.py` | AOT import、ABI/device/hash/symbol 核验；production 禁止自动 JIT |
| `_ext/_streaming_forward_sm90*.so` | 独立 forward AOT module |
| `_ext/_streaming_delta_sm90*.so` | 独立 delta AOT module |
| `_ext/_stable_dkdv_sm90*.so` | 独立 dK+dV AOT module |
| `_ext/_fused_dq_dbias_sm90*.so` | 独立 dQ+dBias/reduce AOT module |
| `streaming_triangle_forward_v1.{py,cu}` | online-softmax forward 与 LSE |
| `streaming_triangle_delta_v1.{py,cu}` | backward delta 预计算 |
| `stable_triangle_dkdv_v2.{py,cu}` | generic/runtime dK+dV |
| `fused_triangle_dq_dbias_v9.{py,cu}` | fixed-tree dQ+dBias |

上述四组 CUDA 源作为 build receipt/source distribution 进入 package-data，
但 production runtime 只 import wheel 内四个 `.so`，不调用 NVCC。只有开发者
显式设置 `OPENFOLD3_TRIANGLE_ALLOW_JIT=1` 时 loader 才允许 source JIT；缺失、
损坏或 ABI 不匹配的生产二进制会直接报错。生产 facade 不允许依赖
`scripts/benchmarks`、旧 resident dK/dV 或旧 v7 backward loader。

发布 wheel ABI 固定为 CPython 3.14、Torch 2.10.0、Torch CUDA 12.9 与
H100 `sm_90` SASS；不包含 PTX。build manifest 必须记录 commit、dirty 状态、
ABI/toolchain、所有生产源 SHA-256 以及四个 `.so` 的 SHA-256 和导出符号。
canonical builder 必须使用 fresh clone，并通过 `nm`、`cuobjdump` 和只读、
无 NVCC clean-install 的 public-facade F+B smoke。

## 数值契约

Lambda Streaming 与官方 PyTorch 使用相同实数公式，但浮点执行契约不同：

- raw BF16 Q/K 进入 HMMA，FP32 accumulator 后乘 scale；
- mask 与 pair bias 在 FP32 中加入；
- forward 使用在线 softmax；
- backward 由保存的 LSE 重算概率；
- dBias 使用固定归约树，不使用浮点 atomic accumulation；
- query slab、batch chunk 与 leaf wave 只划分互相独立的输出或已定义的
  leaf 区间；跨 wave 的 FP32 root 始终按全局 leaf index 左折叠，因此 workspace
  计划不会改变归约树；
- workspace planner 的默认目标是 256 MiB（目前不是公开配置项）。若一个
  最小 `batch x 32-query x leaf` tile 本身超过目标，允许使用该不可约下限。

因此当前承诺是**同后端、同输入、同环境下的逐 bit self-replay**，而不是对
PyTorch、AQLab 或 cuEq 的逐 bit 一致。训练可用性必须由长期 trajectory、
optimizer/EMA 状态、validation quality 与 time-to-quality 裁决，不能仅由一次
operator replay 推断；这些长期证据目前尚未完成。

## 非生产与历史实现

`pytorch_aligned_*`、`stable_triangle_attention.py`、resident dK/dV、v7/v8
backward 以及 `scripts/benchmarks` 中的裸 kernel 链都是参考、诊断或历史实现。
当前 Lambda 模型路由不会调用它们；observer 仍可监视 aligned facade，目的只是
证明其调用次数为零。

性能数字只有在 clean commit、真实 non-contiguous Q/K/V、permuted BF16
pair bias、官方 mask、统一 downstream consumer、平衡执行顺序及 raw sample
回执全部具备时才能写入本文件。开发期 smoke 数字不得升级为发布结论。
