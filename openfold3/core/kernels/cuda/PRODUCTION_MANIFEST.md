# Lambda Streaming Triangle Attention — release manifest

本文件描述当前由 `use_stable_triangle_attention=true` 选择的 Lambda
Triangle Attention 路径。OpenFold3 默认配置仍保持官方实现；Lambda 路径必须
显式启用，并且对不支持的输入直接报错，不会静默回退 PyTorch、AQLab Triton
或 NVIDIA cuEquivariance。

## 当前发布状态

**状态：production candidate（尚未 production-ready）。**

当前已经具备：

- package-owned public autograd facade 与 CUDA 源码；
- online-softmax forward；
- zero-atomics recompute backward；
- 同一输入的逐 bit self-replay；
- CUDA device、dtype、shape、grid 与 stream 的 fail-closed 检查；
- generic runtime sequence/head 路径与非整 tile 尾块保护；
- H100 Compute Sanitizer component smoke。

正式放行仍要求同一 clean commit 上完成：

1. public-facade 四后端公平性能报告；
2. AQLab Triton、NVIDIA cuEq、Lambda Streaming 的 20-step 训练保险丝；
3. 三后端 1000-step fresh-process stability soak；
4. 路由、初始状态、batch 顺序、源码/环境哈希和训练状态完整回执。

## 运行时支持范围

- GPU：Hopper `sm_90`（当前发布门禁设备为 Lambda H100 80GB）；
- dtype：Q/K/V 与 pair bias 为 BF16；mask 为 FP32/BF16/FP16；
- head dimension：`D in {16, 32, 64, 128}`；
- sequence/head：设备可行范围内的 runtime `N > 0`、`H > 0`；
- CUDA grid 边界：`N <= 65535`、`H <= 65535`，实际可用上限还受
  `O(N^2)` 显存与运行时间限制；
- batch prefix：任意非空、可广播的正 prefix；当前实现逐 leaf 调度，功能泛化，
  但尚不是 batch-prefix 并行吞吐优化。

`N=384/640/768` 是编译期快车道，不是功能白名单。其他长度走
`StaticSequence=0` generic kernel，并使用显式尾块 guard；不会因为长度不同而
回退其他后端。

## 生产调用链

| 文件 | 职责 |
|---|---|
| `openfold3/core/model/primitives/attention.py` | 显式 Lambda 路由；unsupported fail-closed |
| `streaming_triangle_attention.py` | public facade、broadcast/prefix 编排、自定义 autograd |
| `streaming_triangle_forward_v1.{py,cu}` | online-softmax forward 与 LSE |
| `streaming_triangle_delta_v1.{py,cu}` | backward delta 预计算 |
| `stable_triangle_dkdv_v2.{py,cu}` | generic/runtime dK+dV |
| `fused_triangle_dq_dbias_v9.{py,cu}` | fixed-tree dQ+dBias |

上述四组 CUDA 源必须进入 wheel package-data。生产 facade 不允许依赖
`scripts/benchmarks`、旧 resident dK/dV 或旧 v7 backward loader。

## 数值契约

Lambda Streaming 与官方 PyTorch 使用相同实数公式，但浮点执行契约不同：

- raw BF16 Q/K 进入 HMMA，FP32 accumulator 后乘 scale；
- mask 与 pair bias 在 FP32 中加入；
- forward 使用在线 softmax；
- backward 由保存的 LSE 重算概率；
- dBias 使用固定归约树，不使用浮点 atomic accumulation。

因此当前承诺是**同后端、同输入、同环境下的逐 bit self-replay**，而不是对
PyTorch、AQLab 或 cuEq 的逐 bit 一致。训练可用性由 20/1000-step 轨迹、
optimizer/EMA 状态和验证质量裁决，不能仅由一次 operator replay 推断。

## 非生产与历史实现

`pytorch_aligned_*`、`stable_triangle_attention.py`、resident dK/dV、v7/v8
backward 以及 `scripts/benchmarks` 中的裸 kernel 链都是参考、诊断或历史实现。
当前 Lambda 模型路由不会调用它们；observer 仍可监视 aligned facade，目的只是
证明其调用次数为零。

性能数字只有在 clean commit、真实 non-contiguous Q/K/V、permuted BF16
pair bias、官方 mask、统一 downstream consumer、平衡执行顺序及 raw sample
回执全部具备时才能写入本文件。开发期 smoke 数字不得升级为发布结论。
