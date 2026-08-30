# StableFold Triangle Attention — release manifest

本文件描述当前由 `use_stable_triangle_attention=true` 选择的 StableFold
Triangle Attention 路径。OpenFold3 默认配置仍保持官方实现；StableFold 路径必须
显式启用。公开 facade 对不支持的输入直接报错；模型级 blanket-coverage 路由会对
不满足 facade 契约的站点记录一次原因并回退官方 PyTorch attention。

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

- GPU：Hopper `sm_90a`（当前发布门禁设备为 H100 80GB）；
- dtype：Q/K/V 为 BF16；pair bias 接受 BF16 或 FP32，FP32 在 facade 的
  Hopper 内核边界显式转为 BF16（其梯度由 autograd 转回 FP32）；mask 接受
  FP32/BF16/FP16；
- head dimension：`1 <= D <= 64`；8 的倍数原生走 io-dim 车道，其余维度
  zero-pad 到下一个 8 的倍数并保持有效区域梯度；
- sequence/head：设备可行范围内的 runtime `N > 0`、`H > 0`；
- CUDA grid 边界：`ceil(N/32) <= 65535`、`H <= 65535`、flatten 后
  `B*N <= 2^31-1`；实际可用上限还受 `O(N^2)` 输入/输出显存与运行时间限制；
- batch prefix：任意非空、可广播的正 prefix 会 flatten 为一个 CUDA batch，
  不再逐 leaf Python 调度；
- pair bias：若确实需要 broadcast，target prefix rank 必须 `<=16`；若 bias
  prefix 已与 target prefix 完全一致，则没有这一 broadcast bookkeeping 限制。

生产 facade 对所有 sequence 统一 pad-to-64，并走 `StaticSequence=-1` 的
runtime exact-tile lane；没有 `N=384/640/768` 或 `N=640` 的分发表。只有绕过
facade 的未填充 direct ABI 调用才走 `StaticSequence=0` 的显式尾块 guard。

## 生产调用链

| 文件 | 职责 |
|---|---|
| `openfold3/core/model/primitives/attention.py` | 显式 StableFold 路由；unsupported 模型站点回退 stock |
| `stablefold_triangle_attention.py` | public facade、broadcast/prefix 编排、自定义 autograd |
| `_aot_runtime.py` | AOT import、ABI/device/hash/symbol 核验；production 禁止自动 JIT |
| `_ext/_stablefold_forward_sm90a*.so` | forward 与 fused physical-layout QKV pack AOT module |
| `_ext/_stablefold_delta_sm90a*.so` | 独立 delta AOT module |
| `_ext/_stablefold_dkdv_sm90a*.so` | 独立 dK+dV AOT module |
| `_ext/_stablefold_dq_dbias_sm90a*.so` | 独立 dQ+dBias/reduce AOT module |
| `stablefold_triangle_forward.{py,cu}` | fused QKV pack、online-softmax forward 与 LSE |
| `stablefold_triangle_delta.{py,cu}` | backward delta 预计算 |
| `stablefold_triangle_dkdv.{py,cu}` | generic/runtime dK+dV |
| `stablefold_triangle_dq_dbias.{py,cu}` | fixed-tree dQ+dBias |

上述四组 CUDA 源作为 build receipt/source distribution 进入 package-data，
但 production runtime 只 import wheel 内四个 `.so`，不调用 NVCC。只有开发者
显式设置 `OPENFOLD3_TRIANGLE_ALLOW_JIT=1` 时 loader 才允许 source JIT；缺失、
损坏或 ABI 不匹配的生产二进制会直接报错。生产 facade 不允许依赖
`scripts/benchmarks` 或任何历史 backward loader。

共享 AOT wheel 还包含独立的 `_stablefold_trimul_sm90a` 模块；它复用同一
manifest、ABI/hash preflight 与显式 development-JIT gate，但其数值和支持范围
不属于本 Attention release manifest。

发布 wheel ABI 固定为 CPython 3.14、Torch 2.10.0、Torch CUDA 12.9 与
H100 `sm_90a` SASS；不包含 PTX。build manifest 必须记录 commit、dirty 状态、
ABI/toolchain、所有生产源 SHA-256 以及全部五个 `.so` 的 SHA-256 和导出符号。
canonical builder 必须使用 fresh clone，并通过 `nm`、`cuobjdump` 和只读、
无 NVCC clean-install 的 public-facade F+B smoke。

## 数值契约

StableFold 与官方 PyTorch 使用相同实数公式，但浮点执行契约不同：

- raw BF16 Q/K 进入 HMMA，FP32 accumulator 后乘 scale；
- mask 以 FP32 加入；pair bias 从 BF16 内核存储提升到 FP32 后加入；
- forward 使用在线 softmax；
- backward 由保存的 LSE 重算概率；
- dQ/dBias 使用重算的 `T=sum(P*dP)` 行标量；基于实测
  `sum(P)=1 ± 1.3e-6`，当前执行契约省略 `T/Z` 的除法；
- dQ/dBias 的每个 CTA 将四个 outer 共用的 `32x32` BF16 bias tile 只读取
  一次，存入 Q shared tile 未参与 MMA 的 `+8` padding；下一 tile 的 bias 与
  各 warp 私有 K/V 在现有 dBias 尾部 barrier 前预取，不增加 dynamic shared
  memory，不改变浮点运算或归约顺序，也没有 sequence 特判；
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

`pytorch_aligned_*`、`stable_triangle_attention.py`、resident dK/dV、历史
backward 以及 `scripts/benchmarks` 中的裸 kernel 链都是参考、诊断或历史实现。
当前 StableFold 模型路由不会调用它们；observer 仍可监视 aligned facade，目的只是
证明其调用次数为零。

性能数字只有在 clean commit、真实 non-contiguous Q/K/V、permuted BF16
pair bias、官方 mask、统一 downstream consumer、平衡执行顺序及 raw sample
回执全部具备时才能写入本文件。开发期 smoke 数字不得升级为发布结论。
