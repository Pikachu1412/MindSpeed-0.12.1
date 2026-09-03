# 流式词表并行交叉熵

## 功能概述

流式词表并行交叉熵将 GPT 输出层的词表投影（Linear）和逐 token Cross Entropy 合并处理，并沿 token 维分块计算。训练过程中不再物化完整的 `[tokens, local_vocab]` logits，从而显著降低输出层和交叉熵的峰值显存。

该功能通过 MindSpeed patch 接入 Megatron MCore GPT，无需修改 Megatron 源码，并保持输出为 `[batch, sequence]` 的逐 token loss。

## 使用方法

在 GPT 训练参数中增加：

```bash
--use-streamed-vocab-parallel-cross-entropy \
--streamed-vocab-parallel-cross-entropy-chunk-size 4096
```

建议 chunk size 使用 **2048 或 4096**，并保持为 8 的倍数，以启用单次 fused backward 快速路径。

## 新增能力

- 支持 FP16、BF16 训练。
- 支持 tensor parallel、context parallel 和 sequence parallel。
- 支持任意逐 token upstream gradient，语义与 Megatron 原始 vocab-parallel CE 一致。
- 使用现有 torch-npu/CANN 融合算子完成：
  - 分块词表投影和在线 max/sum 统计；
  - 基于统计量的交叉熵计算；
  - hidden gradient 和 weight gradient 计算。
- TP 通信按统计量批量执行：forward 仅进行一次 MAX 和一次打包 SUM all-reduce。
- 功能为显式 opt-in；未开启时完全保留原始 Megatron 路径。
- 启动时检查所需 torch-npu 算子及 schema，不兼容时提前报错。

## 本轮优化

### 1. backward 复用 forward 统计

forward 保存 global max、global sum-exp、target mask 和 masked target 等 O(tokens) 紧凑统计。backward 不再重复执行分块投影和 TP 统计通信。

效果：

- `fused_linear_online_max_sum` 调用数由 `2 × chunks` 降为 `chunks`；
- TP2 backward 不再产生重复的 CE MAX/SUM collective；
- 保留 hidden-gradient 所需的正常 TP 通信。

### 2. 单次 all-token fused backward

当 chunk size 是 8 的倍数时，将 packed target metadata 合并，整个 token batch 只调用一次 fused backward kernel，而不是每个 chunk 各调用一次。

非 8 对齐 chunk 保留兼容 fallback，但性能和低精度 weight-gradient 累加稳定性不如对齐路径，因此不建议使用。

### 3. 紧凑保存 TP reduction 结果

对 global sum-exp 进行独立物化，避免 tensor view 持有完整的 `[2, tokens]` packed reduction storage。

### 4. benchmark 和回归测试增强

- benchmark 支持 TP1 和 TP2；
- baseline 与 streamed 使用一致的 TP dgrad 语义；
- correctness gate 覆盖随机、带符号、非均匀逐 token gradient；
- 新增测试确保 backward 不重算 forward 统计；
- 新增测试确保对齐 chunk 只调用一次 fused backward；
- 显式拒绝不兼容的 `--unaligned-linear` 配置。

## 性能结果

测试环境：Ascend 910B、BF16、tokens=4096、hidden=1024、global vocab=32768。

| 场景 | baseline | streamed | CE 增量峰值显存 |
|---|---:|---:|---:|
| TP1 | 13.04 ms | 16.13 ms | 1024 MiB → 442 MiB（约 -56.8%） |
| TP2 | 8.35 ms | 9.95 ms | 512 MiB → 282 MiB（约 -44.9%） |

端到端 TP2 GPT mock-data 测试：

- 普通词表（vocab=32768）：baseline 与 streamed 基本持平；
- 大词表（vocab=131072）：streamed 中位迭代时间约慢 4.9%；
- CE 局部显存下降明确，但完整模型总峰值还会受到参数、优化器、激活及 allocator cache 影响。

因此当前功能定位是：**显著降低 CE 峰值显存，并将端到端性能控制在基本持平到约 5% 回退范围内**，暂不宣称稳定加速。

## 当前限制

仅支持 MCore GPT 训练路径，并要求：

- `parallel_output=True`、输出层 `gather_output=False`；
- 输出层无 bias；
- `label_smoothing=0`；
- 不开启 deterministic mode；
- 不与 cross-entropy fusion、gradient-accumulation fusion 同时使用；
- 不支持 unaligned linear、deferred embedding wgrad、MTP 和 config logging；
- 每个 TP rank 的 local vocabulary 至少包含 128 项；
- 当前不支持带 labels 的 evaluation/inference。

另外，ModelOpt distillation、legacy model、空 token 输入以及部分依赖普通 Tensor logits 的 hook 尚未纳入支持范围，启用前应结合实际训练配置验证。

## 相关代码

- 实现：`mindspeed/core/tensor_parallel/streamed_vocab_parallel_cross_entropy.py`
- 特性注册：`mindspeed/features_manager/tensor_parallel/streamed_vocab_parallel_cross_entropy.py`
- 单元测试：`tests_extend/unit_tests/features/tensor_parallel/test_streamed_vocab_parallel_cross_entropy.py`
- 性能测试：`tests_extend/benchmarks/benchmark_streamed_vocab_parallel_cross_entropy.py`
