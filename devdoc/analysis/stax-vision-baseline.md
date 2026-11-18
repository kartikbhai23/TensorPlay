# Stax 视觉模型编译性能基线

- 对象：`tensorplay.compile(backend="stax")` 编译 `tensorplay.vision.models` 全部可实例化分类模型
- 基线：参考运行时的编译执行（逐模型同模式、同权重）与两侧的 eager 执行
- 口径：稳态推理延迟、单步训练延迟、首次编译到首个完成的调用
- 设备：CUDA（RTX 3090）为主，CPU 同步覆盖
- 工具：`benchmark/bench_stax_vision.py`
- 基线日期：2026-11-17

本轮定位的是“差距在哪里”，不是最终结论。下表所有数字均可由仓库内脚本复现。

## 0. 结论

**CUDA 上 stax 的 GPU 计算时间已与参考编译执行基本持平，差距几乎全部来自每次调用的宿主侧开销，以及卷积算法选择。**

| 模型 | stax GPU ms/次 | 参考编译 GPU ms/次 | stax 墙钟 ms | 参考编译墙钟 ms |
| --- | ---: | ---: | ---: | ---: |
| resnet18 | 6.55 | 6.71 | 3.90 | 3.63 |
| resnet34 | 6.60 | 7.11 | 8.48 | 7.29 |
| mobilenet_v2 | 2.62 | 2.53 | 3.88 | 2.82 |

`GPU ms/次` 为 profiler 统计的 10 次迭代总核时间均值，`墙钟` 取 `bench_stax_vision.py`（16 批、224 输入、10 次中位数）的同一轮结果。GPU 时间互有胜负，但墙钟一致落后，说明瓶颈在“把核送进队列”与“决策”。

CPU 上差距是另一个量级：MobileNetV2 在 8 批、224 输入下 stax 1835.7 ms、参考 eager 2072.0 ms、参考编译 132.6 ms。参考编译的收益来自按通道序布局与算子级融合，本仓库 CPU 代码生成尚未覆盖这一层。

因此“全部超越”目前不成立。已修复的是编译失败与每调用状态分析；尚未完成的是宿主派发、卷积算法选择、CPU 算子融合三块系统性工作。

## 1. 判定口径与复现

```bash
# CUDA，推理 + 训练 + 编译耗时
python benchmark/bench_stax_vision.py \
  --batch 16 --tasks infer,train \
  --json-out benchmark/results/stax-vision-cuda.json

# 只看推理（全模型）
python benchmark/bench_stax_vision.py \
  --batch 8 --tasks infer \
  --json-out benchmark/results/stax-vision-cuda-all.json

# CPU
python benchmark/bench_stax_vision.py \
  --device cpu --batch 8 --tasks infer \
  --json-out benchmark/results/stax-vision-cpu.json

# 编译成本：两侧都禁用可读缓存
python benchmark/bench_stax_vision.py \
  --batch 8 --tasks infer --cold-compile \
  --json-out benchmark/results/stax-vision-cold.json
```

脚本对每个模型做三件事：

1. 用同一份参数分别构建两侧模型（严格 `load_state_dict`），先校验 eager 与编译结果的数值一致；
2. 同模式编译（`fullgraph=True`；stax 侧 `strict_native=True`），记录“请求编译 → 首个调用完成”的墙钟；
3. 分别测 eager 与编译后的稳态推理、单步训练（模型编译，损失/反向/优化器在外）。

未编译成功或数值不一致的用例记为 `error`，不参与加速比。

## 2. 已修复项

### 2.1 视图未落盘导致卷积无法降级（P0）

`convnext_tiny`、`shufflenet_v2_x1_0` 等模型在 `strict_native` 下直接失败：

```
realize NYI on ir.Pointwise
While executing %conv2d_0 (args = (%permute_0, ...))
```

`lower_permute`/`lower_slice`/`lower_reshape` 等构造视图时把持有值的盒子剥掉，只留下尚未落盘的循环体。卷积入口要求输入在内存中，只能顺着视图向下请求内存，而裸循环体没有这条路径。

修复：视图在源值已落盘时直接持有缓冲，否则保留持有它的盒子（与调度自身的 `make_view` 规则一致）。提交 `4c121a4a`。修复后两个模型均可编译并通过数值校验。

### 2.2 每调用重算“捕获依赖的模块状态”（P1）

编译产物每次调用都要判断：模块层级、训练开关、子模块绑定，以及捕获时被固化进程序的全局/闭包/模块属性是否变化。原实现对每个模块重新推导“运行哪些函数、读哪些属性名”，对 MobileNet 量级模型每次约 1.7 ms，超过其 GPU 核时间的一半。

修复：

- 运行函数集合按模块类缓存，属性名集合按类缓存；
- 函数读取的全局名与闭包槽位按代码对象缓存；
- 不可变值按对象身份缓存其冻结描述；
- 每次调用的“读取站点”按程序构建一次计划，只有在可达模块/函数集合变化时重建。

提交 `00017795`。同一模型的状态分量从 1.72 ms 降到 0.81 ms（MobileNetV2），ResNet18 从 0.68 ms 降到 0.32 ms，测试 `test/stax/test_stax_captured_values.py`、`test/stax/test_guards.py`、`test/stax/test_region_cache.py` 全部通过。

## 3. 剩余差距与根因

### 3.1 宿主侧派发（CUDA，全部模型）

编译产物是生成的 Python wrapper：每次调用要重算所有输入的断言、创建中间缓冲、逐个发起核。对 ResNet34 一次调用约 36 个卷积、67 个 Triton 核、183 个输入断言。t 态持续吞吐为 6.51 ms（GPU 6.60 ms），但单次 event 墙钟为 7.81 ms，说明首个核之前有约 1 ms 的串行宿主工作。

可量化的部分：

| 分量 | ResNet34 每次调用 |
| --- | ---: |
| 模块状态分量 | 0.32 ms |
| `boxed_call` + 生成 wrapper | 约 0.7 ms |
| 卷积派发（36 次） | 约 0.35 ms |
| Triton 启动（67 次） | 约 0.2 ms |

改进方向：把模块状态比对下沉到已有 C 派发器（`_stax_native.make_call_dispatcher` 已暴露 fast/fingerprint 通道）；生成 wrapper 的输入断言与参数整理按编译期计划折叠；静态形状的推理路径默认录制 CUDA 图（当前树式录制在动态分配/拷贝入出上仍有约 1 ms 固定成本，尚未净收益，需要继续压缩）。

### 3.2 卷积算法选择（CUDA，卷积类模型）

cuDNN 启发式选择不是每个形状的最优解。实测把 `tp.backends.cudnn.benchmark` 打开后：

| 模型 | 关闭 | 打开 |
| --- | ---: | ---: |
| resnet18 | 4.18 ms | 3.67 ms |
| resnet34 | 7.24 ms | 6.71 ms |
| mobilenet_v2 | 3.83 ms | 3.92 ms |

ResNet 系收益 7%–12%，深度可分离网络可能变差。做成编译器能力而非全局开关：编译期为每个卷积形状在真实布局下试算候选算法并固定结果，记录试算成本；深度可分离卷积保留启发式。参考实现的 `max-autotune` 只试 Triton 模板，不试框架卷积算法，这是可以正面超过它的点。

### 3.3 权重布局转换（CUDA，全部卷积模型）

生成代码在每次调用里把每个卷积权重转换到通道序（`triton_poi_fused_*` 拷贝），ResNet34 一次调用 36 次。权重只随参数版本变化，可按 `(存储指针, 版本)` 跨调用缓存转换结果，并在版本变化时失效。收益随权重量占比上升，VGG/RegNet 系更明显。

### 3.4 CPU 算子融合（CPU，全部模型）

CPU 侧参考编译把深度可分离卷积与逐点算子保持在同一通道序布局并融合，MobileNetV2 从 2072 ms 降到 133 ms。本仓库 CPU 路径逐算子调用，卷积走通用实现，没有等价的布局与融合层。这不是局部调优能补的差距，需要一条独立的 CPU 融合后端（通道序布局传播 + 卷积/逐点融合模板 + 线程切分）。

### 3.5 编译耗时

当前基线里编译耗时读数大量为 0.00 s，因为两侧都有磁盘缓存命中。`--cold-compile` 会同时关闭两侧的产物/内核缓存（stax 的 `config.force_disable_caches`、参考侧的 `force_disable_caches`）并在每个用例前重置编译状态；脚本已分别记录“创建编译对象”与“首次调用”两段。

## 4. 阶段计划

按“先让所有模型可编译并数值正确，再逐项拿回性能”的顺序：

| 阶段 | 内容 | 预期 | 状态 |
| --- | --- | --- | --- |
| P0-1 | 视图落盘、全模型可编译 | 消除 error 用例 | 已完成 |
| P0-2 | 模块状态分析缓存 | 每调用 −0.4~−0.9 ms | 已完成 |
| P1-1 | 模块状态比对下沉 C 派发器 | 每调用再 −0.3~−0.8 ms | 待做 |
| P1-2 | 编译期卷积算法试算与固定 | 卷积模型 −5%~−12% | 待做 |
| P1-3 | 权重布局转换跨调用缓存 | −1%~−5% | 待做 |
| P1-4 | 生成 wrapper 参数整理折叠 | −0.3~−0.7 ms | 待做 |
| P1-5 | CUDA 图录制固定成本压缩 | 小模型 −10%~−30% | 待做 |
| P2-1 | CPU 通道序布局 + 卷积/逐点融合 | MobileNet 级需 10x 量级 | 待做 |
| P2-2 | 编译耗时对比闭环（双侧禁缓存） | 判定编译成本 | 待做 |

每一阶段完成后用同一脚本重跑并更新本文件；只有 `vsCompiled ≥ 1.0` 且数值校验通过的用例才计入“已超越”。
## 5. 训练步的图回放（2026-11-18）

`mode="reduce-overhead"` 下前向与反向各自是一段被录制的区域。前向区域写归一化统计，因此按“写输入的录制”处理：绑定调用方自己的存储，不暂存参数。反向区域不写输入，走直接绑定。

### 5.1 只暂存真正移动的输入

直接绑定读的是调用方存储，所以一个每次调用都换地址的输入会让整个区域退回全量暂存：训练一步里 120 余个输入（参数、保存值）各自一次拷贝。实测反向区域只有**一个**位置每步漂移（logits 梯度），其余位置稳定。

改为让漂移异常带上漂移位置，包装器只把移动的位置改读私有缓冲、其余继续读调用方存储；只有全部输入都移动时才退回全量暂存，写输入的区域永暂存。展开视图（步长为 0）没有合法私有缓冲：拷进稠密缓冲会给录制一个核并未按其编译的布局，因此这类输入在两种管理器里都保持绑定。

| 模型（批 8，训练一步） | 之前 stax | 之后 stax | 参考编译 | 占比 |
| --- | ---: | ---: | ---: | ---: |
| resnet18 | 9.02 | 8.27 | 7.11 | 0.86 |
| vit_b_32 | 21.29 | 20.66 | 17.88 | 0.85 |
| densenet121 | 27.73 | 25.85 | 19.27 | 0.75 |
| googlenet | — | 14.82 | 11.45 | 0.77 |

resnet18 的 `Memcpy DtoD` 从 0.881 ms/步 降到 0.219 ms/步。

### 5.2 训练侧剩余差距的成分（GPU 设备忙时间，批 8）

resnet18：stax 8.25 ms / 参考 6.92 ms，按类归因（+1.33 ms）：

| 类别 | stax | 参考 | 差 |
| --- | ---: | ---: | ---: |
| 卷积/矩阵乘 | 4.36 | 3.87 | +0.49 |
| Triton 逐点/归约 | 1.74 | 1.51 | +0.23 |
| cuDNN 布局转换 | 1.44 | 1.09 | +0.35 |
| 框架原生核（softmax/nll/pool/sgd） | 0.46 | 0.00 | +0.46 |
| 设备内拷贝 | 0.23 | 0.00 | +0.23 |

其中 cuDNN 布局转换的 24 次/步来自反向：模板给反向卷积要了通道序，而 cuDNN 为该描述符选的仍是行主序引擎（`wgrad_alg0_engine`），于是自己在内部打包（`nhwcToNchwKernel`）；参考侧选到的是 `wgrad_alg0_engine_NHWC`（0.293 ms 对 0.346 ms）。`TP_STAX_CONV_CHANNELS_LAST=0` 下 resnet18 训练降到 7.64 ms（设备忙 6.81，已低于参考的 6.93），但 googlenet 训练从 14.8 升到 16.5——布局取向必须按区域测量，不能改成全局默认。

vit_b_32 的差距是另一组成分：

| 项 | 每次 | 说明 |
| --- | ---: | --- |
| 反向输出拷贝 | 1.05 ms | 权重梯度克隆（25×9.4 MB + 76×3 KB） |
| 原生逐点（`binary_same_shape_vectorized`） | 1.31 ms | 原生注意力反向内部的元素算子 |
| 原生行广播/逐元素 | 0.46 ms | 同上 |

参考编译这两项都是 0：它把残差加法融进注意力尾声，输出直接交给调用方（自己的池做 live 记账）。

### 5.3 下一步

1. 反向卷积的布局请求按区域测量：同一描述符下行主序引擎与通道序引擎各自的总时（含 cuDNN 内部转换），择优固定；这是 resnet 系 −0.9 ms 的来源。
2. 反向输出不再每步克隆：需要在池上做 live 记账（录制输出可被后续回放复用而不失效），否则调用方跨步持有会静默看到新值。
3. 原生注意力反向内部的逐点算子合并；残差加法融进注意力尾声。
4. densenet121 的 Triton 启动数 1421/步（参考 679），逐点融合链仍不足——这是小通道卷积网络的共性差距。
5. 全量 sweep 时注意：某用例的内存压力会让下一个用例的 `cuBLASLt` 工作区在捕获窗口内分配失败（`cuBLASLt Error 13`），该区域随即退回未录制执行（vit_b_32 一次 23.2 → 21.3 ms）。

## 6. 反向卷积的梯度布局（2026-11-18）

`conv2d_grad_weight_cudnn_v8` 原来固定用行主序描述符建 filter 梯度：算子是通道序时，cuDNN 仍选行主序引擎（`wgrad_alg0_engine`），于是在**每次读算子**时把通道序的激活/梯度输出重排一遍（`nhwcToNchwKernel`）。改为梯度描述符跟随算子布局（K-R-S-C），结果再 repack 回行主序——参数住在行主序。开关 `TP_CONV_BWD_FILTER_ROW_MAJOR=1` 回到旧行为，用于同一次构建上的 A/B。

| 模型（批 8，训练一步，设备忙 ms） | 旧行为 | 现在 | 参考编译 | 占比 |
| --- | ---: | ---: | ---: | ---: |
| resnet18 | 8.25 | 7.74 | 6.92 | 0.89 |
| resnet50 | — | 18.28 | 16.92 | 0.93 |
| googlenet | — | 11.20 | 9.93 | 0.89 |
| densenet121 | — | 23.87 | 18.47 | 0.77 |

resnet18 的 `nhwcToNchwKernel` 24 次/步随之消失（0.289 → 0.004 ms/步），代价是 3 个 filter 大小的 repack（`Memcpy DtoD` 9.4/2.4/0.6 MB 各一次）。

### 6.1 densenet121 的差距离谱（设备忙，批 8）

| 类别 | stax | 参考 | 差 |
| --- | ---: | ---: | ---: |
| 卷积/矩阵乘 | 9.75 | 7.31 | +2.44 |
| Triton 逐点/归约 | 10.11 | 6.77 | +3.34 |
| cuDNN 布局转换 | 2.07 | 0.67 | +1.40 |
| 框架原生核 | 1.07 | 0.00 | +1.07 |
| 设备内拷贝 | 0.70 | 0.00 | +0.70 |

- `convertTensor_kernel` 240 次/步、1.92 ms：通道序卷积 plan 自带的一次重排（输入行主序时换成 `nchwToNhwc` + `nhwcToNchw`，共 0.017 ms/次；两端都是通道序时只有这一个，0.020 ms/次）。plan 计时是含这次重排的端到端时间（`TP_CONV_PLAN_DEBUG` 的 0.112 ms 与实测 0.111 ms 相符），所以这不是“选错引擎”，而是该形状在通道序下的下限。
- Triton 里 `triton_per_fused_*` 459 次/步（参考 99 次）：densenet 有 120 个 BN，每个 BN 的反向被展成 4 个小核（两个梯度归约 + 两个逐点）。
- 设备内拷贝 489 次/步，来源已定位。`input_update_placeholders` 只有 121 个（每个 BN 的 `num_batches_tracked`），其余是**梯度累加**：区域把每个参数的梯度算在自己的缓冲里，再由引擎拷进 `.grad`（densenet 120 个 BN 的 weight+bias 各一次，加上卷积权重）。resnet18 同样：20 次 `num_batches_tracked` + 40 次 BN 参数梯度 + 4 次 147 KB + 3 次 filter 梯度 repack。参考侧为 0：它的图直接写 `.grad`。
- 试过的替代方案（未采纳）：让训练 BN 直接走框架原生 `batch_norm` / `batch_norm_backward`（`remember_batch_statistics` 已在）。数值更准（单步梯度相对误差 1.2e-6，Triton 分解是 1.2e-3），但训练一步：resnet18 7.74 → 7.54、googlenet 11.20 → 11.38、resnet50 18.28 → 18.40、densenet121 23.87 → 25.06——通道序下原生核反而更贵，且它自己写 running stats 与“区域不写输入”的约定冲突。代码已回退。

### 6.2 下一步

1. 梯度累加不再每步拷贝：区域把梯度算在自己的缓冲里，引擎再拷进 `.grad`。需要在池上做 live 记账（录制输出可被后续回放复用而不失效），否则调用方跨步持有时会静默看到新值；这是 densenet121 上最大的一块（约 0.7 ms/步）。
2. input_updates 的 20~121 次 8 字节拷贝（`num_batches_tracked`）：合并成一个多张量拷贝核。
3. 卷积 plan 的候选集：densenet 里 `cutlass__5x_cudnn` 48 次/步 1.77 ms，参考 38 次 0.51 ms，说明有些形状可选的引擎集合里没有参考侧那一个——值得按形状核对 INSTANT 列表与 `drop` 过滤。
4. 推理侧（批 8，`--mode reduce-overhead`）现状：convnext_tiny 9.21 vs 6.79（0.74）、swin_t 10.24 vs 7.87（0.77）、maxvit_t 13.44 vs 13.06（0.97）、efficientnet_b7 13.28 vs 14.16（1.07）。convnext 的差距集中在 18 个 depthwise 卷积（`conv2d_grouped_direct_kernel` 1.19 ms，参考的同名族 0.61 ms）：把 depthwise 也改成通道序（`TP_STAX_CONV_CHANNELS_LAST_MODE=depthwise`）反而更慢（9.21 → 13.93 ms），说明该模式下权重没有真的按通道序传递，cuDNN 每次调用都重排。
5. 其余同 5.3 的第 2、3、5 条。
