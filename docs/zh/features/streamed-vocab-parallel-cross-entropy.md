# 流式词表并行交叉熵

## 功能概述

流式词表并行交叉熵将 GPT 输出层的词表投影（Linear）和逐 token Cross Entropy 合并处理。forward 由 CANN 算子在词表维内部流式计算在线统计；大 token batch 的 backward 在 MindSpeed 中按 local vocab 分块。训练过程中不再物化完整的 `[tokens, local_vocab]` logits，从而显著降低输出层和交叉熵的峰值显存。

该功能通过 MindSpeed patch 接入 Megatron MCore GPT，无需修改 Megatron 源码，并保持输出为 `[batch, sequence]` 的逐 token loss。

## 使用方法

在 GPT 训练参数中增加：

```bash
--use-streamed-vocab-parallel-cross-entropy \
--streamed-vocab-parallel-cross-entropy-chunk-size 4096
```

该参数当前仅为命令行兼容项；forward 和 backward 的分块策略由实现根据算子特性和 token 数自动选择。

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

### 1. 单次 all-token fused forward

forward 仅调用一次 `fused_linear_online_max_sum`。算子在 CANN kernel 内沿词表维分块并在线更新 max/sum，不输出完整 logits，因此不再需要 Python token chunk loop、list 和 `torch.cat`。

### 2. backward 复用 forward 统计

forward 保存 global max、global sum-exp 和 target metadata 等 O(tokens) 紧凑数据。backward 不重复执行 CE 统计投影，也不产生重复的 TP MAX/SUM collective。

### 3. 自适应低显存 backward

- token 数小于 2048 时继续使用单次 CANN memory-friendly fused backward，减少小矩阵和 Python 调度开销；
- token 数不小于 2048 且 local vocab 不小于 4096 时，按最多 4096 个 local-vocab entry 分块重建概率，并用矩阵乘分别计算 hidden gradient 和对应的 weight-gradient slice；
- vocab tile 会按 128 MiB 的 BF16/FP16 logits 加 FP32 probability 临时存储预算随 token 数自动缩小，不物化完整 local logits；
- hidden gradient 在 FP32 accumulator 不超过 128 MiB 时跨 tile 使用 FP32 累加并最终转换回输入 dtype；更大形状回退到输入 dtype 累加，避免 accumulator 本身导致 OOM；
- 保留任意带符号、非均匀逐 token upstream gradient 语义。

### 4. 紧凑保存 TP reduction 结果

对 global sum-exp 进行独立物化，避免 tensor view 持有完整的 `[2, tokens]` packed reduction storage。TP forward 仍仅进行一次 MAX 和一次打包 SUM all-reduce。

### 5. benchmark 和回归测试增强

- benchmark 支持 TP1 和 TP2；
- correctness gate 覆盖 FP16/BF16、随机逐 token gradient、TP2+CP2 和 sequence parallel；
- 测试确保 forward 只调用一次 fused statistics kernel；
- 测试分别覆盖小 batch fused backward 和大 batch vocab-tiled backward；
- 显式拒绝不兼容的 `--unaligned-linear` 配置。

## 性能结果

测试环境：Ascend 910B、BF16、tokens=4096、hidden=1024。以下为当前 microbenchmark 的总 forward+backward 延迟和 CE 增量峰值显存：

| 场景 | baseline | streamed | CE 增量峰值显存 |
|---|---:|---:|---:|
| TP1，vocab=32768 | 12.51 ms | 9.25 ms（约快 26.1%） | 保持低于完整 logits 路径 |
| TP2，vocab=32768 | 8.47 ms | 6.49 ms（约快 23.4%） | 512 MiB → 176 MiB（约 -65.6%） |
| TP2，vocab=131072 | 27.49 ms | 21.59 ms（约快 21.5%） | 2048 MiB → 296 MiB（约 -85.5%） |

其中 TP1 拆分计时为：

- baseline：forward 6.36 ms，backward 6.15 ms；
- streamed：forward 1.75 ms，backward 7.49 ms。

与上一版单次 memory-friendly fused backward 相比，vocab-tiled backward 显著缩短了大 token batch 的反向耗时，使该功能从“以速度换显存”变为代表 shape 下同时降低延迟和显存。实际端到端收益仍取决于模型中 CE 的占比、词表大小、TP 切分和 allocator 状态。

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
