# 自动激活内存策略：使用与开发说明

当前实现已整合进MindSpeed仓库。综合优化结果、测试范围、失败边界和性能数据统一见项目根目录`/home/shensy/project/PROJECT_REPORT.md`；本文件仅保留接口与运行约束。

## 单开关入口

在正常模型、数据、batch、精度、并行、设备和优化器配置之外，启用`AUTO_ACTIVATION_MEMORY=1`，或等价训练参数`--auto-activation-memory`。无需另外填写每层KEEP/OFFLOAD列表、checkpoint列表或预取深度。

```bash
cd /home/shensy/project/Megatron-LM-0.12.1
AUTO_ACTIVATION_MEMORY=1 WANDB_MODE=disabled bash run_offload.sh
```

该命令不是固定形状/卡数的benchmark命令；保持正常训练资源配置，并确认PYTHONPATH指向当前MindSpeed仓库。原启动脚本不由本feature修改。

## 策略及执行范围

- 按已支持的Transformer层实例发现attention、norm、Dense/MoE MLP，并产生同一联合plan。
- KEEP保留激活；OFFLOAD使用原saved-tensor运行时；RECOMPUTE执行保留RNG的真实checkpoint。
- Dense MLP支持合法内部activation子边界，可选父KEEP/子重算，避免不必要的整MLP重算。父重算时子KEEP不绕过父边界。
- 同一个求解器选择搬运方式、预取与staging额度。原小张量过滤、参数storage排除和必要事件依赖不变。
- 只管理激活，不卸载参数、梯度或优化器状态。设备Adam照常创建和更新。

旧手动offload入口为兼容保留，不应与AUTO手工叠加另一套activation checkpoint策略。optimizer swap/CPU optimizer、FP8、compile/图捕获等组合不属于当前已支持验收范围。

## profile、缓存与保护

冷启动采用全RECOMPUTE bootstrap：通常2步warmup、4步profile、2步无采样基线，第9步开始执行计划。缓存放在训练cwd的`.mindspeed/activation_profiles/`；所有rank完整兼容才命中。缓存只复用测量，仍重新采集容量并求解，不复用固定plan。

签名包括模型/输入/运行配置和实际源码依赖，包含已加载的linear兼容实现与feature注册源码。读取不了已加载依赖时失败关闭。带宽校准必须保存/恢复RNG，不污染训练随机状态。

预算区分KEEP storage、offload copy字节、checkpoint输入、workspace、并发microbatch及搬运缓冲。保留原安全余量和运行时guard。host-cost反馈、transfer-cost反馈和周期profile不是同一个时钟；重求解完成后才消费成熟反馈。

静态Adam容量明显不足、没有安全可行解、profile自身OOM或运行时容量突变，不能保证无缝继续。当前不提供任意半步失败恢复，不应绕过分布式一致性继续更新。完整checkpoint重启恢复和任意模型支持仍待实现。

## 仓库内回归

依赖与正常训练环境一致，需要CANN/NNAL环境及可导入的Megatron。CPU验收不启动训练，不初始化NPU：

```bash
cd /home/shensy/project/MindSpeed-0.12.1
PYTHONPATH=/home/shensy/project/MindSpeed-0.12.1:/home/shensy/project/Megatron-LM-0.12.1 \
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1 \
/home/shensy/envs/mindspeed/bin/python tests_extend/run_auto_activation_memory_cpu_tests.py
```

可选`--output /absolute/path/result.json`将结果写到一个尚不存在的文件。测试位于`tests_extend/unit_tests/mindspeed/core/pipeline_parallel/`，覆盖联合求解、父子边界、生命周期、搬运、显存计数、RNG、反馈、缓存和linear上下文。测试和运行器不依赖已删除的benchmark源码路径。

2026-09-22落仓检查为354项通过；随后已直接使用原始run_offload.sh完成Dense12/S4352的80步cold和20步warm原生训练，未使用源码覆盖或测试bootstrap。三种策略、周期profile、真实缓存命中和显存门槛通过；cold/warm日志有小幅数值差异，未采完整梯度哈希，不能当作新增的逐元素梯度一致性证明。这也不替代完整Dense32B/MoE30A3B或任意模块验收。

本轮专用程序为`tests_extend/run_auto_activation_memory_native_smoke.py`，使用原脚本启动上述两个case，并提供`--validate-only`及`--audit-only`。它不自动删除缓存或制造签名；cold前提需要按实际缓存状态确认，不能把已有HIT冒充cold。结果必须写到新的服务器project/reports目录，普通训练仍直接调用原run_offload.sh。
