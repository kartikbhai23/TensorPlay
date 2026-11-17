# Stax 编译器对照上游审计报告

- 审计对象：`tensorplay/compiler/`（含 `backends/stax/`、`_core/`、`backends/`）、`tensorplay/graph/`、`tensorplay/export/`
- 对照基线：仓库内 `third_party/pytorch` 的 `torch/_dynamo`、`torch/_inductor`、`torch/_decomp`、`torch/_functorch/aot_autograd`、`torch/export`
- 审计日期：2026-11-14
- 方式：源码静态审计 + 定向运行复现 + `pytest test/stax` 全量执行
- 结论口径：只把“已验证”的写为事实；静态推断但未运行验证的，单独标注
- 修复状态：本文标注“已修复”的项已落到代码，修复内容见 §13

---

## 0. 结论摘要

**核心中游（降级、调度、代码生成）与上游结构基本对齐；主要缺口在前端捕获模型、联合图 pass 仍未接线、以及部署/量化/多设备等边缘子系统。**

按优先级：

| 级别 | 问题 | 性质 | 状态 |
| --- | --- | --- | --- |
| P0 | `dynamic=True` 时非张量标量参数被陈旧工件复用，结果错误 | 正确性 bug，已复现 | 已修复 |
| P0 | 数据相关输出形状参与 Python 控制流时无守卫，静默算错 | 正确性 bug，已复现 | 已修复 |
| P0 | 上游 pass 管线（约 1.1 万行）未接入主路径 | 能力缺失，已核实 | 已接 post-grad + 联合图 |
| P0 | `stax.fusion` / `stax.triton` / `stax.max_autotune` / `stax.coordinate_descent_tuning` 选项被接收后丢弃 | 静默失效，已核实 | 已修复 |
| P0 | 四个环境变量设置被同文件后段的硬编码值覆盖 | 静默失效，已核实 | 已修复 |
| P1 | post-grad 缺 `reorder_for_locality` / `remove_assert_ops` / `reciprocal_sqrt_to_rsqrt`，`split_cat` 表无人调用 | 能力缺失，已核实 | 已补 |
| P1 | `record_function` span 不进图，因此 post-grad 的 span 删除规则没有作用对象 | 能力缺失，已核实 | 已对齐（§4.6） |
| P1 | 切分与合并互为逆运算时两个算子都不必留下，此前只有点名 `split_cat` 表才处理 | 能力缺失，已核实 | 已补（§4.2） |
| P1 | 乘积与其加数分两次调用；分解成三节点链的 `addcdiv` 失去单次调用的融合 | 能力缺失，已核实 | 已补（§4.2、§13） |
| P1 | 前端无字节码帧求值/图中断/恢复，Python 副作用只发生一次 | 架构差异，已复现 | 未修（架构选择） |
| P1 | `dynamic_shapes` 只接受 bool，无自动 dynamic、无 `Dim`、`mark_static` 为空操作 | 能力缺失，已核实 | 已修复（§3.2、§13） |
| P0 | 反向扫描按输入值聚合贡献，`x*x`、`x+x`、两路汇到同一值时只保留一项，梯度错误 | 正确性 bug，已复现 | 已修复（§3.4、§13） |
| P0 | 部分归约的梯度按输入整形状回铺：`x.sum(1)` 直接报错，某些轴组合会铺到错轴 | 正确性 bug，已复现 | 已修复（§3.4、§13） |
| P1 | 区域内创建的常量被提升为模块属性，反向却把它当参数向调用者索要，`KeyError` | 正确性 bug，已复现 | 已修复（§3.4、§13） |
| P1 | 多输出区域的联合输出把第二个结果当成梯度：min-cut 直接抽取失败，结构分区只在无需保存时碰巧正确 | 正确性 bug，已复现 | 已修复（§3.4、§13） |
| P0 | `call_method` 形式的原地写不被算作写，min-cut 把 `buf.add_(1)` 丢掉，值与梯度都是写前的 | 正确性 bug，已复现 | 已修复（§3.4、§13） |
| P2 | `tp.compile(name=...)` 传给每个后端后无人读取，区域标识没有任何作用 | 静默失效，已核实 | 已修复（§5、§13） |
| P2 | `tp.compile(isolate_recompiles=...)` 一路传递并存在调用对象上，从无读者；两种取值结果相同 | 静默失效，已核实 | 已拒绝（§5、§13） |
| P1 | `write_are_deterministic_algorithms_enabled` 的读者存在、写者缺失，要求确定性算术时归约仍可能选到自由排序的配置 | 正确性相关，已核实 | 已修复（§8.5、§13） |
| P2 | `use_launch_metadata_schema` 设置了也没用，启动器仍按对象上可读的字段探测 | 静默失效，已核实 | 已修复（§8.5、§13） |
| P2 | `export` / `unused` / `overload_method` 文档暗示影响捕获，实际只是写标记；`substitute_in_graph` 的两个参数一个只存不读、一个只是提示 | 声明与实际不符，已核实 | 已澄清（§8.4、§13） |
| P2 | `export(strict=...)` 接受两种取值后 `del` 掉，docstring 自称两模式一致 | 静默失效，已核实 | 已拒绝（§8.4、§13） |
| P0 | 捕获时读到的非张量全局/闭包/模块属性改变后仍复用旧程序，结果陈旧 | 正确性 bug，已复现 | 已修复（§3.3、§13） |
| P1 | AOT 的 `memory_budget` 与 `keep_inference_input_mutations` 接收后丢弃 | 静默失效，已核实 | 已修复（§3.4、§13） |
| P1 | 无 AOTInductor 部署包、无 oneDNN/MKLDNN/quantized/freezing | 子系统缺失，静态确认 | 部署包已实现（CPU C++ wrapper，§6、§13），已覆盖全连接与卷积网络；C++ wrapper 本身此前无测试，且原地复用、视图、非缓冲参数三处都读错/写不出（§8.7），已修；oneDNN/MKLDNN/quantized/freezing 未修 |
| P1 | cudagraph 只有单层录制：输入一换地址就回退到未捕获工件，输出被下次回放覆盖 | 能力缺失，静态确认 | 已实现树式录制：按布局分节点、共享一个内存池、copy-in/copy-back（§8.2、§13） |
| P1 | 无 CPU GEMM 模板；`config.cpp.max_autotune_gemm_backends` 里的 "CPP" 无人消费 | 能力缺失，静态确认 | 已实现 `CppGemmTemplate`（f32/bf16/f16，静态 M/N/K，不预打包权重） |
| P1 | 设备只注册 cpu/cuda；MPS/XPU/MTIA/ROCm 无代码生成 | 设备覆盖缺口，静态确认 | 未修 |
| P1 | 分布式 overlap、bucketing 未接线 | 运行时能力缺口，静态确认 | 未修 |
| P1 | 工件缓存只存打印好的 wrapper，且要等降级、调度、打印都做完才发现命中 | 能力缺失，静态确认 | 已实现整图缓存：键在降级之前定下，命中即跳过降级/调度/打印/构建（§8.1、§13） |
| P2 | 多处死 import 会在特定路径抛 `ImportError` | 移植残留，已核实路径 | 已清理 |
| P2 | 移植模块读取的设置名在本模块不存在，特定机器/配置上才走到就 `AttributeError`；`tp_configs` 这个隔离键形式也会撞上同一缺口 | 静默缺陷，静态确认 | 已修复（§8.5、§13） |
| P2 | 配置面数字过期（190 项 / 1,793 行） | 一致性问题 | 已更新为实测 195 项 / 1,894 行（§1） |

`pytest test/stax` 实测（修复前）：**1568 passed, 2 failed, 1 skipped**（1571 项，约 2 分 4 秒）。
其中一个失败是确定性的 P0 正确性 bug；另一个是容差/顺序相关的偶发（单独运行通过）。

`pytest test/stax -q -p no:randomly` 实测（本轮修复后）：**1849 passed, 1 skipped**（1850 项，约 23 分钟）。
此前一轮曾记录 `1806 passed, 5 failed`：5 项失败里 3 项是工作区里进行中的前端/内核改动（`upsample_nearest2d`
同时收到尺寸与缩放、`one_hot` 走回退），已分别修掉（§13），剩下 2 项是随机顺序导致的偶发，会话级
缓存隔离后不再复现（§8.3）。
`pytest test/fx test/test_public_surface.py test/higher_order_ops test/export`：**376 passed**。
原先那两个失败都已不再是失败：确定性那个随 §2.1 修好；偶发那个是 bf16 容差写死 `2e-4` 造成的，
现在按 dtype 的机器精度取容差。

---

## 1. 规模与结构基线

| 维度 | 本仓库 | 上游对照 |
| --- | --- | --- |
| 编译器 Python 总量 | `tensorplay/compiler/` 190,685 行（stax 181,912） | `torch/_inductor/` 284,829 行 |
| 图/前端 | `tensorplay/graph/` 35,577 行 | `torch/_dynamo/` 130,241 行 + `torch/fx/` |
| stax 后端文件数 | 239 个 `.py` | 379 个 `.py` |
| post-grad pass | `graph_passes/post_grad.py` 378 行 | `fx_passes/post_grad.py` 2,521 行 |
| 配置面 | `stax/config.py` 1,920 行 / 195 顶层项 | 3,204 行 / 306 顶层项 |

核心文件行数并不落后，部分还略多：

| 文件 | 本仓库 | 上游 |
| --- | ---: | ---: |
| `ir.py` | 12,660 | 12,373 |
| `scheduler.py` | 11,396 | 11,387 |
| `op_lowerings.py` / `lowering.py` | 10,738 | 9,747 |
| `graph_lowering.py` / `graph.py` | 2,994 | 3,240 |
| `codegen/triton.py` | 9,167 | 8,788 |
| `codegen/cpp.py` | 7,198 | 6,487 |
| `codegen/wrapper.py` | 5,319 | 5,274 |
| `pattern_matcher.py` | 3,117 | 3,087 |

因此，“对比上游是否还有不完善的地方”的答案不是“核心编译器没搬完”，而是**前端、pass 接线与边缘子系统**。

---

## 2. P0：正确性问题

### 2.1 `dynamic=True` + 标量参数：陈旧工件复用

**现象**

```python
import tensorplay as tp

def f(x, s):
    return x * s

c = tp.compile(f, backend="stax", dynamic=True)
x = tp.randn(3, 4)
assert tp.equal(c(x, 2), x * 2)   # 通过
assert tp.equal(c(x, 3), x * 3)   # 失败：仍按 s=2 计算
```

- 关键字参数、仅关键字参数、位置标量均可复现。
- `dynamic=False` 正常；`backend="eager"` 正常。
- 现有测试 `test/stax/test_stax_symbolic_shapes.py::test_dynamic_capture_binds_keywords_and_default_arguments` 因此失败（`test_stax_symbolic_shapes.py:230`）。

**根因（已定位）**

`stax` 的图工件缓存 `graph_artifact_cache.graph_key()` 只把**带 `.shape` 的输入**计入键：

```python
# tensorplay/compiler/backends/stax/graph_artifact_cache.py
inputs = [
    _tensor_spec(tensor)
    for tensor in getattr(graph, "example_inputs", ())
    if hasattr(tensor, "shape")
]
```

标量输入（`2` / `3`）不进入键。而降低阶段会把标量样例值直接写进生成的 wrapper/内核（实测生成的 C++ 内核中出现 `static_cast<float>(2.0)`），于是第二次编译命中第一次的工件，返回按 `s=2` 计算的结果。

**验证**

- 把 `tensorplay.compiler.backends.stax.config.force_disable_caches` 设为 `True` 后，同一用例 `s=3` 正确。
- 说明不是前端捕获错误（图里确实是 `placeholder s`，输入样例是 `3`），而是工件键遗漏。

**修复方向**

1. `graph_key` 对**所有** `example_inputs` 生成摘要，而不只张量：
   - 标量：类型名 + `repr(value)`（区分 `2` 与 `2.0`、`True` 与 `1`）；
   - 容器：递归；
   - 未知对象：类型名 + `repr`（宁可键不稳定导致 miss，也不能碰撞导致错）；
   - 张量：保留现有 `_tensor_spec`。
2. 提升 `SCHEMA_VERSION`，让旧条目自然失效。
3. 增加回归测试：同一函数在 `dynamic=True` 下用不同标量值连续调用，断言都正确。

**风险**：低。只影响缓存命中率；键变长/变严会多编译，不会算错。

**修复结果（已落地）**

- `graph_key()` 改为对**全部** `example_inputs` 取摘要：张量沿用 `_tensor_spec`；`float` 用 `float.hex`、`complex` 用实部虚部 `float.hex`（避免 `repr` 的精度歧义）；`bool`/`int`/`str`/`None` 用 `repr`；序列与映射递归；其余对象用类型名 + `repr`；深度上限 `_MAX_ARG_DEPTH`。
- `SCHEMA_VERSION` 提升到 `"2"`，旧条目自然失效。
- 回归测试：`test/stax/test_stax_symbolic_shapes.py::test_scalar_arguments_are_part_of_the_cached_artifact`；原有 `test_dynamic_capture_binds_keywords_and_default_arguments` 恢复通过。

---

### 2.2 数据相关输出形状参与控制流：静默错

**现象**

```python
def f(x):
    y = x[x > 0]          # 输出长度取决于数据
    return x + 1 if y.numel() > 0 else x - 1

c = tp.compile(f, backend="stax")
c(tp.ones(4))    # -> [2,2,2,2] 正确
c(-tp.ones(4))   # -> [0,0,0,0] 错误；期望 [-2,-2,-2,-2]
```

`y.numel()` 在捕获时是普通 Python `int`，分支被直接烘焙；缓存键只覆盖输入形状，不覆盖数据决定的输出长度，因此第二次调用不会重编译，也没有守卫。

**机制（已核实）**

- 捕获期 `x[x > 0]` 记录为 `operator.getitem`，随后 `y.numel()` 返回具体整数，`if` 在 Python 层完成。
- 死代码消除后主图只剩 `add`（条件子树不可达被删除），没有任何关于 `y` 长度的守卫。
- 文档 `devdoc/stax-chapter-split/stax-05-cache-measure-exercises.md:655` 已承认：`nonzero`/`unique`/`bincount` 走回退内核，依赖其大小的形状“只能按具体值特化”；符号捕获入口不支持数据相关输出尺寸。
- 但**静态捕获路径没有拒绝或守卫**，于是从“不支持”变成了“静默算错”。

**修复方向（建议按序）**

1. **短期（安全优先）**：识别数据相关尺寸的生产者（布尔 mask 的 `getitem`、`nonzero`、`masked_select`、`unique*`、`bincount`），当程序读取其 `numel` / `size` / `shape` 并用于 Python 控制流时，抛出带解释的 `GraphCaptureError`，而不是烘焙。
2. **中期**：把这些读取变成数据门（复用 `data_specializations` + `_extract_guard_replay`），运行时重算尺寸并比较；尺寸变化触发重编译。注意 `shape` 返回的是 tuple，需要可追踪的 tuple 或在 `size(i)` 处埋点。
3. **长期**：无提示符号 + 运行时断言，或对该类区域直接回退 eager（图中断语义）。

**测试**：新增“布尔 mask 后读 `numel` 分支”的正反用例；断言要么正确重编译，要么明确抛错，不能返回错误值。

**修复结果（已落地）**

审计时只发现“分支被烘焙”，复核时发现同一根因还带**越界读写**，比原判更严重：

```python
def f(x):
    return x[x > 0].unsqueeze(0) * 2      # 消费者按追踪样例的 2 个元素定尺寸
```

在 `x` 全为负时按 0 个元素产出长度 2 的结果，`index*2` 甚至读到缓冲区之外（进程退出时 `munmap_chunk(): invalid pointer` / SIGABRT）。`ms.sum` 在 0 个元素时返回垃圾值。

根因链：布尔 mask 的 `index.Tensor` 在 `index_impl` 中抛错 → `fallback_handler` → `FallbackKernel.create` → `process_kernel` 用零填充样例跑一遍得到长度 0 → `_make_fallback` 用**追踪样例**的长度（静态 2）定尺寸。

修复分三层，遵循“宁可不编译，也不要算错”：

1. **捕获期拒绝**：`graph/_utils.py` 新增 `extent_follows_data(target, args, dtype_of)`，识别 `nonzero` / `masked_select` / `bincount` / `unique*` / `repeat_interleave` / 布尔索引的 `index`、`_unsafe_index`、`getitem`；`Tracer` 给这类节点打上 `dynamic_extent_nodes`；`Proxy` 在读取 `shape` / `stride` / `numel` / `nelement` / `len()` 时抛 `GraphCaptureError`，说明“结果元素个数取决于数据，捕获期无法得知”。
2. **降级期拒绝**：`op_lowerings.refuse_sized_from_sample()` / `graph_lowering._make_fallback()` 在该节点有非输出使用者时抛 `NotImplementedError`（`strict_native=True` 时为硬错误）。
3. **其余情况回退**：只把被选中的值当结果返回、不据其定尺寸的图仍然编译。

行为与上游一致（同样的程序在上游也是被拒绝的：`UserError: Could not guard on data-dependent expression`）。

测试：`test/stax/test_compile.py::test_data_decided_length_cannot_be_read_while_capturing`、`::test_operations_sized_from_a_selected_length_do_not_trust_the_example`、`::test_selected_values_returned_as_the_result_still_compile`。

---

## 3. 前端捕获模型 vs 帧求值

### 3.1 架构差异（已复现）

| 能力 | 本仓库 | 上游 |
| --- | --- | --- |
| 捕获方式 | 代理对象在样例上真实执行一遍（`Tracer(execute=True)`） | 字节码帧求值 + 符号执行 + resume |
| 图中断 | 无；整区捕获失败即报错（`_core/api.py:1246-1252`） | 有；可切段、段间交回解释器 |
| Python 副作用 | 只在捕获时发生一次 | 每次调用在图外重放 |
| `fullgraph` | 只影响重编译上限错误（`_core/api.py:603-608`） | 禁止图中断 |
| `*args/**kwargs` | 直接拒绝（`graph/tracer.py:601-608`） | 支持 |
| 数据相关分支 | 按样例特化 + 数据门守卫，分支翻转整区重编译 | 图中断 + resume，或 unbacked SymInt |
| 数据相关输出尺寸 | 静态路径无守卫（见 2.2） | 图中断 / `capture_scalar_outputs` / 断言 |

实测副作用：

```python
calls = []
def f(x):
    calls.append(1)
    return x + 1

c = tp.compile(f, fullgraph=True, strict_native=True)
for _ in range(3):
    c(tp.randn(2, 2))
# len(calls) == 1，而不是 3
```

`fullgraph=True` 也不会阻止这一行为，因为它没有“图中断”可供禁止。

### 3.2 动态形状 API 不完整

- ~~`tp.compile(dynamic_shapes=...)` 只接受 bool；`Dim` / 范围约束只存在于 `tensorplay.export`，未接到 `compile`。~~
  已修复：`dynamic_shapes` 接受按参数名的映射或按位置的序列，`Dim` / `Dim.AUTO` /
  `Dim.STATIC` / 静态 int 都生效；具名轴成符号，未具名轴保留样例尺寸并在运行前校验；
  `min`/`max` 与静态尺寸在捕获前对样例校验；每次调用还校验声明上界、共享维上的
  精确尺寸关系，以及派生维的 `size == scale*root + offset`，不满足就抛
  `ValueError` / `RuntimeError`，不会复用按旧尺寸捕获的区域。需要声明
  `supports_symbolic_shapes` 的后端，否则在 `compile` 时明确报错。三种写法
  （映射、位置、装饰器）都有测试。
- ~~无自动 dynamic：12 个不同 shape 调用 → 12 次后端编译。~~
  已修复：`config.automatic_dynamic_shapes`（默认开）在“仅张量尺寸变化导致守卫失配”
  时清空具体特化并以符号重捕一次；数据分支、梯度态、模块态、dtype/device/layout
  变化仍走普通特化。实测 3 个不同 size → 2 次后端编译（先具体、再符号），后续复用。
- ~~`mark_static` 直接返回输入；`disallow_in_graph` 不做任何登记。~~
  已修复：`mark_static` 在编译前把轴从形状策略中移除，在捕获中钉住样例尺寸并保留
  等式守卫；`disallow_in_graph` 登记被禁目标，捕获适配器和 `Tracer.create_proxy`
  两条路径都会抛 `GraphCaptureError`，不再静默吸收。
- 仍无 `mark_dynamic` / `constrain_as_size` / `mark_unbacked` / `explain` 对应物；
  单区域捕获模型没有图中断，“禁止吸收”只能整段拒绝，不能把前后两段拆成两个图。

### 3.3 守卫覆盖面

- 已覆盖：张量类型、形状/秩、dtype、device、requires_grad、stride、容器内容、Python 标量、梯度模式、模块身份/训练态/子模块绑定、分支结果。
- 已覆盖：捕获会读到的非张量全局、闭包单元与模块属性。取值按内容精确比较（浮点按十六进制，容器按类型与元素），
  可变容器与可调用对象按对象身份比较；自身被代码写回的名字不计（它的值是该次调用的结果而非输入）。
  模块经闭包/全局/子模块/方法/辅助方法到达的，都会被枚举。
- 未覆盖：别名/重复输入（同 storage 的两个参数被当作独立输入）、张量数据（数据门之外）。
  可变容器的原地修改本轮已补：全局/闭包/模块属性里的 `list` / `dict` / `set` 改为按内容快照
  进入特化键（嵌套可变对象与可调用对象仍按身份），`list.append`、`dict[k]=v` 一类的原地编辑
  会改变键并触发重捕；捕获代码自己原地修改的名字（`_CALLS.append(...)` 的 `_CALLS`）不进键——
  它的内容是这次调用的效果而不是输入，与"程序自己赋值的名字不计"同一策略。
- 守卫导出/调试：只有 `_tensorplay_last_recompile_reasons` 与 `TP_LOG_RECOMPILES`，没有可执行的 guard 导出。

### 3.4 AOT autograd

- 真实存在：联合图、`partition_default` / max-flow min-cut（`_core/aot.py:928`）、函数化、Philox 种子的 trace 期固定。
- stax 的默认路径会走 AOT（`_core/api.py:1060-1105`），推理与训练都适用。
- 已修复：`memory_budget` 现在真正约束前向留给反向的激活量。取值按比例校验在 `[0, 1]`：
  `0` 一个不留、把反向读到的前向链克隆进反向图，`1` 保留“运行最省”的那个割，
  中间值在“反向至少要的”与“运行最省”两个候选割里选能装进预算的较大者，都装不进时
  从最小的割里按字节从大到小丢弃（丢弃的值由反向从生产者重算）。越界或非数值直接报错。
- 已修复：`keep_inference_input_mutations` 真正选择输入更新的发生位置。`False` 编译出
  无变异的图，运行后把最终值拷回对应参数；`True` 把写留在图里。两者对调用者结果一致。
  更新被移出图的图模块会记录被写的参数位置，图捕获的跳过判定把它算作“输入被修改”，
  不会把仍会写参数的区域当成纯函数重放。
- 已修复：部分归约（`sum` 带 `dim`/`keepdim`）的梯度先在每个被归约的轴上补一个单例，再按输入形状回铺。
  此前直接按输入整形状回铺，`x.sum(1)` 因最后一维对不上直接报错，`x.sum(0)` 在行数恰好匹配时
  可能静默铺在正确位置，形状不匹配的轴组合则会铺错。目标形状取自输入形状，因为建联合图时结果形状
  还没有描述。
- 已修复：区域内创建的常量是模块属性而不是参数，反向占位符若不对应任何参数，就从同一模块的属性读取；
  此前按参数名向调用者索要，`KeyError`。
- 已修复：多输出区域的联合输出被压平，前向结果的个数同时传给两种分区器。此前个数固定为 1，
  第二个结果被当成梯度：min-cut 抽取前向半张图时直接失败，结构分区在前向恰好不需要保存值时
  因为输出仍是嵌套元组而碰巧正确。
- 已修复：原地写（`buf.add_(1.0)` 这类以方法拼写、或函数拼写的 `copy_`）算作写而不是普通计算。
  写节点的结果通常没人读，只按“结果被谁算出来”做的前向闭包会把它丢掉，于是区域在写前的缓冲上
  运行。写节点现在留在前向里，调用者看到的缓冲与 eager 一致。
  - 反向读的是写前还是写后的值，取决于写的位置：图按顺序执行，写之前的读者看到旧值，写之后的读者
    看到新值。仅当“写之前已有读者”且反向确实读到该输入时，才在写之前做一个副本，反向与其他读者
    读这个副本；所有读者都在写之后的，反向直接读输入本身（即写后的值）。
  - 同一输入在写之前和写之后都有读者时，任何单一取值都会让一侧的梯度用错值，因此明确拒绝，
    要求先做函数化。
  - 写进区域内计算出的值时同样拒绝：把它与读者、反向排序正是函数化要做的事，一次切分做不到。
  - 不写值但确有效果的调用（例如标记 profiler 区间的一对调用）同样留在前向里：它们的结果没人读，
    只按“结果被谁算出来”做闭包会把它们一并丢掉，区间就不再被标记。
- 已修复：反向扫描的每条规则返回“每个使用位置一项”的有序贡献表，再按值累加。
  此前规则用“输入值 → 贡献”的映射，两个操作数是同一节点时后一项覆盖前一项，于是
  `(x * x)` 的梯度是 `x` 而不是 `2x`，两条路径汇到同一值时也丢一项。两个操作数是不同
  输入时看不出差别，所以既有用例全部漏过。
- 缺失/受限：
  - 无 compiled autograd、无 AOT autograd cache；
  - 二阶反向导数显式报错；
  - 不感知用户 `utils.checkpoint` 区域。

---

## 4. 上游 pass 管线：post-grad 已接线，联合图仍缺

这是“代码搬了但没接上”的最大一块。

### 4.1 审计时的事实

- 主路径只 import 了一个 pass：`backend.py` 的 `decompose_auto_functionalized`。
- `graph_passes/` 下其余模块只被彼此 import；`post_grad_passes()`、`joint_graph`、`fuse_attention`、`split_cat`、`pad_mm`、`bucketing`、`group_batch_fusion`、`misc_patterns`、`replace_random`、`dedupe_symint_uses` 在主路径无调用者。
- `graph_lowering.py` 没有调用任何 `PatternMatcherPass.apply`。
- `_sfdp_init()` 只被 `graph_passes/joint_graph.py` 调用，而 `joint_graph.py` 本身无人调用；30 个 SDPA 模式实际不会注册生效。

### 4.2 现在的事实（本轮修复后）

`_lower_stax_region` 在区域函数化之后、`GraphLowering` 之前调用 `post_grad_passes(graph_module, not training, input_device)`，其中按顺序执行：

1. `config.dce` 为真时 `graph.eliminate_dead_code()`（新增开关，默认开）；
2. `config.reorder_for_locality` 为真、且是推理或 `config.reorder_for_locality_in_training` 为真时 `reorder_for_locality(graph)`：把生产者的计算挪到读者前面；随机数算子（`tags` 带 `nondeterministic_seeded`）与输入写回边界不跨（两个开关新增，默认开/关）；
3. `config.post_grad_custom_pre_pass`（新增开关，默认 `None`）；
4. `group_batch_fusion_passes(graph, pre_grad=False)`；
5. `remove_noop_ops(graph)`；
6. `remove_assert_ops(graph)`：删掉 `_assert_tensor_metadata`（它是承诺、不是计算，留着会挡融合）；
7. `pass_patterns` 逐张表 `apply`（新增 `reciprocal(sqrt(x)) → rsqrt(x)`；`pass_patterns[2]` 新增两条无条件重写：`cat` 把一次切分的全部片段按原序拼回时整对删除，`split` 恰好沿合并的接缝切开一个 `cat` 时把每个 `getitem` 直接接到对应的片段上。两条都同时认 `split.sizes` 与 `split_with_sizes`：前端把列表切分记为前者，而移植来的表只写了后者。同一张表还新增两条乘积改写：`mm` 的结果与一个可广播、同类型的加数相加时合成一次 `addmm` 调用；分解成 `div → mul → add` 的 `addcdiv` 在 CUDA/XPU 上重新合成一次调用，单次舍入与 FMA 只有合成后才用得上。加数直接来自捕获且结果全部喂给逐点算子的 GPU 乘积保持不合成，留给乘积收尾的逐点融合。`fuse_attention` 的 SDPA 模式与 `pad_mm` 注册在 `joint_graph.patterns`，只在联合图上生效）；
8. `config.post_grad_fusion_options` 里点名的表逐张 `apply`（`split_cat.py` 的 8 张表；默认空表，因此与上游默认一样不跑，用户点名才跑）；
9. `config.post_grad_custom_post_pass`（新增开关，默认 `None`）。

注册表由 `@init_once_fakemode` 的 `lazy_init(input_device)` 在首次使用前初始化，`input_device` 取自区域的第一个带 `.device` 的样例输入。这里**不再**调 `gm.recompile()`：产物直接读图，重编译只会把一次编译变成两次；`_lower_stax_region` 仅在拆 `auto_functionalized` 之后重编译一次。

随机数改写（`replace_random_passes`）不在 post-grad 里，而在联合图 pass 中（训练区域）；推理区域的随机算子直接走 lowering，不需要把种子/偏移改写成后备形式。

接线时还修掉一个系统性缺口：本仓库的图把可选参数记成位置参数（`cat(x, 1)`、`split(x, [2, 2], 1)`），而移植来的模式按关键字写（`dim=Ignored()`）。旧匹配器在比较位置参数个数时就判失败，于是**所有按关键字写的模式在本仓库从未命中**。现在位置/关键字形状对不上时，先用目标算子的声明把调用绑定成“参数名 → 值”（调用省略的参数取声明默认值），两种写法都能命中；`split_cat` 的 `getitem_split_tp` 原先只认 `split_with_sizes`，而前端把列表切分记为 `split.sizes`，一并补上。

**联合图也已接线**：`stax.partition_fn` 指向 `backend.py` 的 `_partition_with_joint_passes`，它在切分联合图之前先跑 `joint_graph_passes(joint_module, input_device)`；`_core/api.py` 组装 AOT 时优先使用编译器自带的 `partition_fn`，没有则退回 `min_cut_rematerialization_partition`。因此 `early_patterns`、联合图侧的 `pad_mm`/`replace_random`、`joint_custom_pre_pass` / `joint_custom_post_pass` 都会作用到真实训练区域。

接线过程中顺带修掉两处会让它一开就崩的问题：

- `canonicalize_quant_mapping()` 无条件访问 `tp.ops.higher_order.invoke_quant_packed`，而本仓库没有注册这个高阶算子（`InvokeQuant` 同样不存在）。现在两者缺席时该 pass 直接跳过。
- `GraphTransformObserver` 在每次 `__enter__` / `__exit__` 都调 `graph.to_dot()`，而没有任何调用方读这两个 dot 串。pass 接线后它会在每个区域的每个 pass 上跑，等于给编译加了一笔纯开销；现在只保留前后节点名差集。

顺序上，联合图在 `min-cut` 之前、post-grad 在分区之后各自跑一遍，`lazy_init()` 由两边共用（同一个 `@init_once_fakemode` 表），注册只发生一次。

### 4.3 与上游的差距

`graph_passes/post_grad.py` 本轮从 565 行补到 704 行（接线前 378 行），上游 2,521 行。仍缺的代表性 pass：

- ~~`reorder_for_locality`~~、~~`remove_assert_ops`~~、~~`reciprocal_sqrt_to_rsqrt`~~：本轮补齐，见 §4.2；
- `remove_profiler_ops`：本仓库没有 profiler 记录算子（`_record_function_enter/exit` 未注册），补了也是死代码，暂不做；
- `decompose_scan_to_while_loop` / `decompose_map_to_while_loop`
- `fix_auto_functionalized_dtype_views`
- ~~`splitwithsizes_cat` / `cat_splitwithsizes`~~：本轮补齐，见 §4.2；
- ~~`cat_slice_cat`（`cat([a, a[:, :n]], 1)` 折成一次 `cat`）~~：复核时已在（上一轮接线时随 `split_cat` 一起落地，本条原记录过时）；
- ~~`view_to_reshape`（只服务 C++ 部署形态）~~：本轮补齐。`view.default` 的 target 在 post-grad 里改成 `reshape.default`（方法拼写的 `view` 一并改写），降低产物只看到一个形状改变的名字；
- ~~`unfuse_bias_add` / `unfuse_bias_baddbmm`~~：本轮补齐。`addmm` / `baddbmm` 的加数是占位符或常量、各操作数同型同设备、且输出的所有使用者都是逐点算子时，把融合乘积改写成逐点链（`mm → alpha 缩放 → beta*inp → 加`），GPU 上才有意义；`beta=0` 仅在 CUDA 上丢弃加数（非 CUDA 需要 expandability 分析，保持融合）。判定用的设备读自区域的真实设备——模式看到的值是 meta 设备的替身，读不到 kernel 会跑在哪里。`beta=0` 的 expandability 分支与半精度 XPU 特例未移植（后者本仓库无 XPU）；
- ~~`view_to_reshape`~~ / ~~`unfuse_bias_add` / `unfuse_bias_baddbmm`~~：见上，均已补齐；
- ~~`pointless_cumsum`（常量填充的 `cumsum` 写成 ramp）~~：本轮补齐。原判的阻塞（新造工厂算子的 `meta["val"]` 落在 meta 设备上）不靠全图假值刷新解决：上游把 `FakeTensorUpdater.incremental_update()` 接在 pass 管线末尾，其无害前提是图上的假值本就是准确的真实设备 fake tensor，刷新幂等；本仓库的值在此构建上报 meta 设备，无条件刷新的净效果无法与进行中的其他改动区分开（见下）。实际上降级前的 `ShapeProp`（以真设备模式跑）本就会给新造节点填假值，ramp 的节点链（arange→mul→view→expand→to）直接以注册算子拼写构造进图即可，替换掉 fill+scan 两步。若后续接入刷新，应限定在真正新造的节点上，而不是整图。
- ~~addmm / addcdiv FMA 融合~~：本轮补齐，见 §4.2；
- `ConstructorMoverPass` / `move_constructors_to_gpu`
- partitioned scatter、WOQ `concat_linear_woq_int4`、mkldnn grouped GEMM

### 4.4 修复后的用户可见影响

- `remove_noop_ops`、batch fusion、`remove_assert_ops`、按需的 `split_cat` 表现在会作用到真实区域（推理与训练都走 post-grad）。
- `reorder_for_locality` 默认在推理区域生效：生产者被挪到读者之前，随机数算子不挪。实测 `(...).relu()` 与 `(...).sin()` 相加再乘第三条分支时，第三条分支的计算被挪到加法之后；带 `randn` 的同类图里 `randn` 保持原位。
- SDPA（`fuse_attention`，30 个模式）与 `pad_mm` 的决策注册在 `joint_graph.patterns` 里，因此只在**联合图**（训练 / 走 AOT 的区域）上生效；纯推理区域不会命中——这与上游把这两族模式只注册进 joint graph 的做法一致。实测：训练区域可编译并通过；推理区域的手写 attention 数值正确但不会改写成融合注意力。
- 切分与合并互为逆运算的整对（`x.split([...])` 的每个片段按原序 `cat` 回来、`split` 恰好沿 `cat` 的接缝切开）默认就会被删掉，不需要点名；只有**部分**片段的重建才需要用户在 `config.post_grad_fusion_options` 里点名 `split_cat` 表。乱序重拼、位置不同的切分都会保留原样。
- `mm` 的结果与一个可广播、同类型的加数相加时，改写为一次带偏置的乘积调用：中间结果不再存在，GPU 上偏置进入乘积收尾。加数不同型、不可广播、或加法带非 1 的缩放时保持两次调用。GPU 上“加数来自占位符/常量且输出只被逐点算子读”的区域仍由 `unfuse_bias_add` 拆回逐点链；两条规则的判定互斥，同一张图不会来回改写。
- 分解成 `div → mul → add` 的 `addcdiv` 在 CUDA/XPU 上重新合成一次调用：商只舍入一次，乘积与加和在支持 FMA 的设备上一步完成；其它设备、带舍入模式的除法、张量缩放仍保留逐点链。
- 内存受限 `mm` 分解仍未接线。

### 4.5 剩余修复方向

1. 给 AOT 阶段加一个“联合图已就绪”的钩子，把 `joint_graph_passes()` 接进去（先只接不依赖缺失模块的部分：`early_patterns`、`pattern_matcher` 表、`replace_random`）。
2. 按需补 §4.3 的 pass，每补一个用 `test/stax` 回归。

### 4.6 profiler span：捕获、丢弃与开关

上游的处理分三段：Dynamo 侧开关 `capture_profiler_record_function` 默认 `False`；开启时把 `record_function`
块写成进入/退出两类算子节点（`profiler._record_function_enter_new`、`_record_function_exit._RecordFunction`，
另有一个旧拼写 `_record_function_enter`）；inductor 侧 `_remove_profiler_ops` 在 custom pre pass **之后**、
`pattern_matcher` **之前**无条件删掉这三类目标，理由是编译区域对 profiler 就是一次调用，区域内部的 span
不再对应独立的工作；分区器把 span 当作技术性不纯算子跳过，避免进入/退出被切到两侧而不成对。

本仓库的 span 不是算子，而是 `record_function` 上下文管理器直接调用 `_C._profiler_user_begin/end`
（`config/native_functions.yaml` 里没有 `profiler::*` 条目）。本轮按上游语义落地，差异点只有“节点怎么表示”：

| 环节 | 上游 | 本仓库 |
| --- | --- | --- |
| 开关 | `dynamo.config.capture_profiler_record_function = False` | `tensorplay.compiler.config.capture_profiler_record_function = False` |
| 写入图 | 三个 profiler 算子 | 两个 Python 调用节点（`_record_function_enter` / `_record_function_exit`），带 `meta["side_effect"]` |
| 删除时机 | custom pre pass 后、`pattern_matcher` 前 | 同位置（`post_grad_passes`） |
| profiler 视角 | 编译区域内的 span 不再上报 | 同上（实测） |

**关键点：图的记录会发生两次。** 第一次是捕获用户程序（`graph.tracer.Tracer`），第二次是把这个图交给
编译器时在调度层重录（`ProxyTensorDispatchMode` 的 `DispatchTracer`），AOT 的前向/反向 trace 都在第二次里。
第一版只认第一种记录，于是节点在第二次重录时“凭空消失”——重录时 `_record_function_enter` 只走了
`_C._profiler_user_begin` 分支，图上什么都没有。现在 `_record_span_call()` 两种记录都认：有
`get_active_tracer()` 就用 `create_proxy`，否则问 `get_proxy_mode()` 拿 `DispatchTracer.record`。
这样 span 在重录后仍是同一对节点，`remove_profiler_ops` 也才有东西可删。

**实测（`test/stax/test_stax_profiler_spans.py`，6 项）：**

- 推理区域：custom pre pass 看到 `[enter, exit]`，custom post pass 为空，结果正确。
- 训练区域：前向区域带 `[enter, exit]`（span 写在可微计算外面），反向区域没有；两侧 post pass 都为空，
  梯度正确。
- 开关关闭（默认）：任何区域都看不到 span 节点。
- profiler 会话：编译后的调用不再出现该 span 名；不在任何捕获区域里的 span 仍然照常上报。

**没有对齐的部分：** 本仓库没有 profiler 算子，进入/退出节点之间没有“句柄”值传递（上游的 exit 吃一个
`_RecordFunction`），节点只靠图内顺序成对，删除时按出现顺序反序擦除；分区器也没有“跳过 span”的分支，
因为 span 在分区前没有被切成两半的经验事实（上面训练用例即验证）。

---

## 5. 后端选项静默失效

`backend.py` 解析并校验了 6 个选项，但 `_lower_stax_region` 只消费了 `use_cuda_codegen`：

```python
# backend.py:117-132
use_fusion = resolved.get("stax.fusion", True)
use_cuda_codegen = resolved.get("stax.cuda_codegen", False)
use_triton = resolved.get("stax.triton", True)
max_autotune = resolved.get("stax.max_autotune", False)
coordinate_descent_tuning = resolved.get("stax.coordinate_descent_tuning", False)
...
# backend.py:253 只用了：
cpp_wrapper=use_cuda_codegen,
```

后果：

- `stax.fusion=False` 不会关闭融合。
- `stax.triton=False` 不会关闭 Triton。
- `mode="max-autotune"` / `mode="max-autotune-no-cudagraphs"` 实际只打开 cudagraphs（或什么都不开），`max_autotune` / `coordinate_descent_tuning` 没有写入 `stax.config`。
- `stax.cuda_codegen=True` 在 CUDA/Triton 路径上 `strict_native=True` 会直接失败（C++ wrapper 只覆盖 CPU-style schedule，`codegen/cpp_wrapper.py:274-284`）。

**修复方向**：把选项转成一次编译内的配置上下文（避免全局污染），或明确拒绝未实现的选项。不能继续“接受但丢弃”。

**修复结果（已落地）**

- `backend.py` 把选项翻成一次 `config.patch(...)` 上下文，覆盖整个 `_lower_stax_region`：`max_autotune=True`、`coordinate_descent_tuning=True`（`mode="max-autotune*"` 与显式选项两条路都走这里）；`stax.fusion=False` 映射为 `pattern_matcher=False` + `max_fusion_size=1`。
- `stax.triton=False` 现在直接抛 `RuntimeError`：这个后端为带 Triton 的设备只写 Triton 程序，没有第二种形态可退。
- `config.py` 补上 `post_grad_custom_pre_pass` / `post_grad_custom_post_pass` 两个开关（此前 `post_grad_passes` 想读也没有名字可读）。
- 顺带修掉一处同源问题：`config.py` 里 `coordinate_descent_tuning`、`benchmark_kernel`、`benchmark_combo_kernel`、`strict_signed_zero` 各有两次顶层赋值，后一次的硬编码值覆盖了前一次的 `os.environ` 读取，于是 `TP_COORDINATE_DESCENT_TUNING`、`TP_BENCHMARK_KERNEL`、`TP_BENCHMARK_COMBO_KERNEL` 三个环境变量一直被静默忽略。删除重复定义后三个变量恢复生效。

同一类问题在其他后端上是**接受即丢弃**：`eager` / `eager_noexcept` / `eager_debug` / `onnxrt` /
`cudagraphs` 收到 `mode`、`options` 时只记一条 warning 就按原样编译；直接构造的 AOT autograd
后端同样。调用者无法从产物看出请求没有生效。现在改为共享的 `check_backend_kwargs`：

- 是否符号化形状与是否要求原生产物这两项只是描述区域，任何后端都可以收到；区域名不再交给后端，
  因为它标识的是“这次编译调用”，由前端自己兑现（见下）；
- 后端没有 `mode` / `options` 含义的，收到就 `RuntimeError`；
- 选项已经绑定进 AOT autograd 后端内部编译器时，再传一次不算多余（它们已经生效）；
- `cudagraphs` 后端此前把 `dynamic` 丢掉，于是可能把会变化的形状录进 CUDA graph；现在按
  `dynamic=True` 跳过，`strict_native=True` 时报错。

前端自己的两个编译选项此前同样没有效果：

- `name=` 被一路传到后端，后端逐个丢弃，区域标识无人读取。现在按“调试元数据”兑现：
  返回的调用对象上保留 `_tensorplay_region_name`，重编译警告与两条重编译上限错误都用它
  给区域起名（未提供时仍用程序自身的名字），并校验必须是字符串。名字**不**进入任何
  特化键或捕获键，起名不会多编译一次、也不会复用别的区域的产物。名字也不再转发给后端：
  前端既然已经自己兑现了它，再交给后端只是给它一个丢弃的对象，而“收到却不兑现”正是
  `check_backend_kwargs` 要拒绝的；直接给后端传 `name=` 现在会明确报错。
- `isolate_recompiles=` 一路传到 `_core/api.py` 并存在调用对象上，但没有任何读者，两种取值
  编译结果相同。本仓库的特化缓存本就属于各自的调用对象：同一个程序的两个 `compile` 调用
  之间不共享条目，嵌套区域也不会占用外层的条目，条目也不会发布到别处——即“隔离”是本前端
  唯一的行为，而该选项要关闭它的那一边（跨调用共享条目）本捕获模型并不提供。两种取值都
  无法兑现一个真实的开关，因此不再提供该关键字：调用者会拿到指名该参数的错误，而不是一个
  看起来按请求编译好的产物。

---

## 6. 缺失的边缘子系统

| 子系统 | 现状 | 上游对应 |
| --- | --- | --- |
| AOTInductor 部署包 | 已有 `build_aot_package` / `load_aot_package`（`backends/stax/aot_package.py`）：把区域写成 `model.so` + `weights.bin` + `metadata.json` + `aoti_metadata.h` + `aoti_entry.cpp`，入口是 `aoti_entry(inputs, outputs, sizevars)`，常量先经 `aoti_set_constants` 交出；包只需运行期库，不需要 Python。覆盖 C++ wrapper 能写出的 CPU 调度：逐点链与 `mm`/`matmul`/`bmm`/`linear`/`addmm`/`conv2d`（Triton kernel、回退到图执行器的区域、以及返回视图的区域明确拒绝） | 可产出 `.so` + 包，C++ 运行时加载 |
| cudagraph trees | 只有 per-region flat manager，`max_entries=8`；dynamic/训练/原地修改直接 skip（`cudagraphs.py:206,776-777,784`） | 树式复用、跨重编译内存池、多树 replay |
| oneDNN/MKLDNN | 无 `mkldnn_ir.py` / `mkldnn_lowerings.py`；`is_mkldnn_conv` 引用的 pass 不存在（`graph_lowering.py:472-473`） | CPU conv/linear 融合、channels-last、prepack |
| 量化 | 无 `quantized_lowerings.py`；int4 仅 fallback；无 WOQ concat_linear | int8/fp8/mixed GEMM、量化 pattern |
| CPU GEMM 模板 | 有 `CppGemmTemplate`（`codegen/cpp_gemm_template.py`）：AVX2/AVX512 的 micro-gemm（f32/bf16/f16）、线程/缓存分块、偏置与可折叠 epilogue 融进 store；不预打包权重、不做 int8/woq/AMX/Brgemm，动态 M 明确拒绝 | CPU 分块 GEMM、micro-gemm 变体（AVX512/AMX/Brgemm） |
| freezing | `freezing_utils.py` 存在，但 `enter_freezing()` 无调用者；无 `freezing.py` | 推理参数冻结 + prepack |
| jagged/nested | 仅 library-call fallback | nested tensor 原生 lowering |
| ROCm/XPU/MPS/MTIA | 设备只注册 cpu/cuda（`codegen/common.py:741-753`）；无对应 codegen 目录 | 各自 codegen 与 device op overrides |
| 整图 FX cache | `graph_artifact_cache` 缓存整个区域工件：键含捕获程序、每个样例输入、模块状态、全部设置、包版本与工具链，在降级之前定下，命中即跳过降级/调度/打印/构建；符号形状区域不入缓存 | 跨进程复用整张编译图（`FxGraphCache`） |
| 分布式 overlap | `_CollectiveKernel` 从不构造；bucketing pass 无调用者；`reorder_for_compute_comm_overlap` 默认关 | collective lowering、overlap 调度、bucketing、FSDP/DDP 融合 |
| profiler / fuzzer / bisector | 无 | 编译诊断与算子覆盖 fuzz |

---

## 7. 死 import 与移植残留（已清理）

审计发现的 12 处“import 到不存在的模块 / 少一层点”，全部处理完毕。原则：**能实现就实现，不能实现就明确拒绝，不留“一旦开启就崩”的路径**。

| 位置 | 原 import | 触发条件 | 处理 |
| --- | --- | --- | --- |
| `scheduler.py` | `.codegen.xpu.xpu_combined_scheduling` | foreach / combo kernel 节点 | 删掉（本仓库无 XPU codegen） |
| `codegen/triton.py` | `..codegen.xpu_combined_scheduling` | `debug_triton_code` | 删掉 |
| `codegen/triton.py` | `.triton_split_scan` | split scan | 改成明确 `NotImplementedError`（`SplitScan` 目前不会被构造：`Reduction.num_splits` 对 scan 恒返回 1 段，但保留显式拒绝而非误用普通 scan 内核） |
| `codegen/simd.py` | `.triton_combo_kernel` | combo kernel codegen | 删除整条 combo 打印路径（约 480 行，本就不可达），`codegen_combo_kernel` 改为明确 `NotImplementedError` |
| `codegen/subgraph.py` | `....graph.experimental.proxy_tensor` | subgraph 分解 trace | 点层数改为 `.....`，指向 `tensorplay.graph.experimental.proxy_tensor` |
| `utils.py` | `.cudagraphs` | `is_cudagraph_unsafe_op` | 改为 `..cudagraphs`（`compiler/backends/cudagraphs.py`） |
| `fx_utils.py` | `.compile_fx._get_subgraph_names` | `FakeTensorUpdater` | 本地实现 `_get_subgraph_names()`（读 `gm` 上以 `subgraph_` 命名的子图属性，与 `control_dependencies.get_subgraph_name()` 的命名约定一致） |
| `graph_passes/pad_mm.py` | `..kernel.mm_common` | pad_mm | 改为 `..templates.mm_common` |
| `graph_passes/joint_graph.py` | `.auto_chunker` | 联合图 | 开启时明确 `NotImplementedError`（`TP_AUTO_CHUNKER` 默认关） |
| `graph_passes/bucketing.py` | `.fsdp` | bucketing | 就地实现祖先 BFS 回退，函数自包含 |
| `scheduler.py` | `.distributed_autotune` | `TP_DISTRIBUTED_MAX_AUTOTUNE_GEMM=1` | 开启时明确 `NotImplementedError`；`virtualized.py` 里因此从未被写入的 `distributed_autotune_state` 槽一并删除 |
| `scheduler.py` | `.codegen.nv_universal_gemm.NVUniversalGemmCaller` | epilogue 融合评估 | 该类在本仓库不存在，5 处 `isinstance` 恒为假，删除分支（行为等价） |
| `comms.py` | `.comms_debug` | `reorder_iterative_debug_memory_recompute` | 该调试开关是硬编码 `False`，连同两处调试对比分支一起删除 |

清理后对 `tensorplay/compiler/` + `tensorplay/graph/` 做了一次全量静态扫描（按相对/绝对 import 解析模块路径并检查文件存在），只剩一处**有意**的 `try: from .fb.implementations ... except ModuleNotFoundError:` 回退实现。

同一轮扫描还发现 `from . import distributed_autotune` 这种“模块名写在 names 里”的形态，此前的简单扫描漏掉了它——已一并处理。

---

## 8. 运行时、缓存与测试

### 8.1 缓存

- `graph_artifact_cache` 缓存整个区域的工件：键在降级之前定下（捕获程序、每个样例输入、模块状态、全部设置、包版本、编译器与系统指纹），命中即跳过降级/调度/打印/构建；符号形状的区域不入缓存。
- `kernel_cache.cache_key` 键为 backend/entry/options/source，**不含工具链版本或 ISA**。
- `codecache.py` 仍只做 kernel/wrapper 级缓存与编译；整图复用由 `graph_artifact_cache` 负责。
- 捕获缓存 `_core/region_cache.py` 默认关闭（需 `TP_CAPTURE_CACHE=1`）。
- `remote_cache.py` 定义了 `RemoteFxGraphCache` / `RemoteAOTAutogradCache`，但只有 autotune 路径调用 `create_cache`。
- 无 TTL / 容量上限；缓存可无限增长。

### 8.2 cudagraphs

录制有两条路径，由 `triton.cudagraph_trees` 选择（默认开）：

- 树关（`CudaGraphManager`）：按输入布局录制，最多 8 个条目；图直接读调用者存储，地址漂移抛
  `CudaGraphInputDrift`，该次调用走未捕获工件。
- 树开（`CUDAGraphTreeManager`）：每个输入布局一个节点，挂在上一次运行所到节点之下；录制持有
  私有输入副本，回放先把调用者的张量抄进去再启动，因此地址漂移不再是回退理由。所有节点录制进
  同一个分配器池（`_C.graph_pool_handle()`）；回放产生的输出克隆后返回，前一次的结果不会被
  下一次回放覆盖。超过上限按最近最少使用淘汰。

有批量入口时回放走 `stage_and_launch`（一次原生调用完成全部输入拷贝与启动），没有时逐张量
`copy_` 再 `replay`。

`dynamic=True` 跳过；训练（输入 requires_grad）跳过；原地修改/输出别名跳过。独立后端此前只在
stax 的 wrap 路径上检查 `dynamic`，后端入口把该参数丢掉；两条路现在一致。

仍未做：分配器侧 checkpoint（活节点把不再读的中间内存还给共享池、而输出仍可读）、跨重编译复用、
把树接进独立 `cudagraphs` 后端（那里本身只做回放，不含 worker 分配语义）。

### 8.3 测试

```
pytest test/stax -q -p no:randomly
# 修复前：1568 passed, 2 failed, 1 skipped
# 本轮修复后：1849 passed, 1 skipped
pytest test/fx test/test_public_surface.py test/higher_order_ops test/export -q
# 376 passed
```

修复前的两个失败：

1. `test_dynamic_capture_binds_keywords_and_default_arguments` —— 确定性，见 §2.1，已随工件键修复。
2. `test_dynamic_low_precision_variance_uses_wide_accumulation[<lambda>1-dtype1-cpu]` ——
   不是顺序问题：断言里写死了 `atol=2e-4`，而 bf16 的机器精度是 `0.0078125`，1 ulp 的
   舍入差就能越界。改成按 dtype 取容差（`finfo(dtype).eps` 的倍数）后稳定通过。

测试覆盖缺口（与上游 `test/inductor` 对比）：

- 无 `torchinductor_opinfo` 全量算子对拍；
- 无 max-autotune 全量、mkldnn、quantization、AOTI、cudagraph trees、compile_worker、fx_passes、pattern_matcher、select_algorithm、templates、foreach、multi_kernel、pad_mm、freezing、jagged、serialization 的对应测试（本轮补了 `test/stax/test_stax_post_grad_passes.py`，覆盖断言删除、局部性重排、随机算子保持、`rsqrt` 改写、切分与合并互为逆运算时的整对删除及乱序/错位保留；以及 `test/stax/test_stax_profiler_spans.py`，覆盖 span 进图与删除、开关默认关闭、训练区域、profiler 会话可见性）；
- `compile_worker` 本轮补 `test/stax/test_stax_compile_worker.py`（8 项）：定时器静默期只触发一次、
  新事件重置静默期、退出取消待触发；心跳槽位未分配时全程空操作、忙槽位上报阶段与 pid、
  `reset` 回收槽位；进程角色位与父进程变化判定。
- pattern matcher / foreach 分组 / split scan 本轮补 `test/stax/test_stax_pattern_foreach_scan.py`（11 项）：
  图模式改写命中节点、`extra_check` 可否决、关键字模式按签名绑定、未命中与空 pass 不动图、非图输入报错；
  foreach 按源/目标 dtype 与形状分组并用 `-1` 分隔、单组返回 `None`、未提示的符号尺寸报错；
  split scan 把每个归约块键抬到下限、程序维键不动。
- 配置面此前无测试，本轮补 `test/stax/test_stax_config_surface.py`（12 项，见 §8.5）；
- 整图缓存补 `test/stax/test_stax_graph_cache.py`（7 项）：键随程序/输入/模块状态/设置/包版本变化、命中时不降级、符号形状不入缓存；
- CPU 分块乘积补 `test/stax/test_stax_cpu_gemm_template.py`（13 项）：`mm`/`addmm`（行偏置与二维偏置）落成模板 kernel 且与 eager 一致、尾块形状、bf16/f16、其它类型不提供、不说话的后端列表不提供、两种 `mm`/`addmm` 拼写都指向同一 lowering、外部 kernel 名接受目标缓冲、边界符号的尺寸提示、读名不触发 store；
- cudagraph 树补 `test/stax/test_cudagraph_trees.py`（17 项）：按布局分节点、交替布局复用、节点共享池、最近最少使用淘汰、树路径、批量入口与逐张量拷贝两条回放路径、输出独立，以及真实 CUDA 上的换地址回放与跨布局录制；
- 下降路径补一条 `test/stax/test_stax_native_lowerings.py::test_a_call_reads_a_value_that_is_still_being_computed`：`mm(relu(x), W)` / `mm(x + x, W)` / `mm(cat([x, x]), W)` 都真的被下降（断言没有降级），见 §8.7；
- 部署包补 `test/stax/test_stax_aot_package.py`（21 项）：包的文件与元数据、常量烘进权重文件、切片常量按连续布局保存、结果跨调用独立、多输出、scratch 缓冲、拒绝无 C++ wrapper 的区域、拒绝非张量输入、拒绝覆盖已有包、坏元数据被拒、重复构建复用已建库，两层全连接网络（四个常量：两层权重与两个偏置）、`mm` 接逐点与“矩阵乘后加偏置”的包、两层卷积网络（三个常量：卷积权重、偏置、第二层权重）、拒绝返回视图的区域，以及两条“不需要编译器”的验证——一个只用 `ctypes` 与 `json` 的子进程直接调 `aoti_entry`，一个用 `g++` 编译、只包含生成头并 `dlopen` 模型的程序；
- C++ wrapper 本轮补 `test/stax/test_stax_cpp_wrapper.py`（34 项，此前该路径无任何测试）：逐点链、多输出、常量、`mm`/`matmul`/`bmm`/`linear`（带与不带偏置）、`addmm`、`conv2d`（无偏置/带偏置/padding 与 stride/后接逐点/两层卷积网络）、两层全连接网络、“外部 kernel 之后接逐点”的整族与逐点链/归约，以及 8 项视图（转置/reshape/切片结果、切片偏移、转置输入、同一缓冲的两个视图、输入自身的视图）；每个用例都断言没有发生降级（§8.7）；
- 无 fuzzer / `test_operators`。

### 8.4 文档声明 vs 实际

| 声明 | 位置 | 实际 |
| --- | --- | --- |
| `reduce-overhead` 折叠启动开销 | 白皮书 `08c:52` | 仅无梯度/无原地/非动态/≤8 布局时；树式录制下输入漂移靠拷贝吸收，输出各自独立（§8.2） |
| Triton 内核并入缓存条目 | `config.py:1310` | `triton_bundler.py` 无调用者 |
| 计算/通信 overlap + bucketing | `config.py:1710`；`config_comms.py` | 配置存在，IR/调度未接线 |
| `NOT_COVERED` 算子“到达即抛” | `operator_coverage.py:15` | 实际走 eager extern fallback；`index_add`/`index_copy`/`scatter_add` 等 6 个已有 lowering 的名字已从表里删除，声明的可覆盖集不再与实表矛盾（§13） |
| stax 后端 173,141 行 | 白皮书 `08c:9` | 当前 181,912 行 |
| `compiler.export` / `unused` / `overload_method` 的文档暗示影响捕获 | `compiler/__init__.py` | 三者只写一个标记，没有读者；`export`/`unused`/`overload` 本是脚本化（scripting）侧的注解，本仓库没有那条路，捕获照常追踪被标注的函数。文档已改写，并指出“不进图”应使用 `disallow_in_graph` |
| `substitute_in_graph(can_constant_fold_through=...)` 像是折叠开关 | `compiler/__init__.py` | 本仓库捕获本来就在追踪期求值“全常量实参”的调用，不会为它建节点；两个取值给出同一张图，参数按提示处理而不是模式开关。`skip_signature_check` 只在注册时生效，因此不再存进记录 |
| 符号捕获不支持无存储伪张量与数据相关输出尺寸 | `docs/source/stax.md:55` | 属实，且应明确到静态捕获路径 |
| `export(strict=...)` 两种取值给出同一张图 | `export/_trace.py:1304` | 属实：捕获只有“记录一次执行”这一条路，且它会拒绝字节码求值路径本该跨过的构造。现按“只接受 `strict=False`；`strict=True` 明确拒绝并说明原因”处理，非 bool 直接报类型错误 |

---

### 8.5 设置面：读取名必须存在

上游 `_inductor/config.py` 是一个扁平模块，各 pass 直接按名字读。本仓库把一部分设置收进
子配置对象（`triton` / `cuda` / `cpp` / `tp_distributed_optimizations` …），移植时名字跟着
旧结构写，于是**代码在、名字不在**：只有走到那条 branch 的机器或配置才会 `AttributeError`。

本轮对 `stax/` 下全部 `config.<name>` 读取做静态核对（含 `if`/`try`/`with` 体的绑定），修掉 5 个
真实缺失名：

| 设置 | 读取方 | 语义 |
| --- | --- | --- |
| `static_launch_user_defined_triton_kernels` | `runtime/triton_heuristics.py` | 用户自带内核是否也走磁盘二进制静态启动 |
| `intra_node_bw` / `inter_node_bw` | `comm_analysis.py` | 单机卡间 / 跨机带宽（GB/s，单向），`None` 表示按卡代次推断 |
| `cudagraph_unsafe_unbacked_ops` | `scheduler.py` | 结果携带数据符号的算子名，其输出符号不可再重放 |
| `save_config_portable` | `runtime/caching/context.py` 经 `tp_configs` | 把设置写成可跨进程比较的文本快照 |

另外两处不是缺名而是读错结构：`bucketing._default_bucket_mode` 读了不存在的
`aten_distributed_optimizations`；`_RuntimeContext.cuda_matmul_precision_configs` 读了本构建
未暴露的 `tp.backends.cuda.matmul.fp32_precision` 与两个缩减开关，导致 `_isolation_key()` 直接抛错。

快照的规则：每个子配置压平成 `section.name`，因此两个 section 可以同名；只保留能写成文本的值
（数、串、布尔、`None` 及其容器），跨进程比较才有意义；设置里的可调用对象写成全限定名，换掉它会
改变快照；本模块自己的辅助函数不算设置；私有名默认不写。

快照 298 项，含私有名 303 项，均可 `json.dumps`。

反方向的核对（定义在 `config.py`、却没有任何读者）又发现三个“设置了也没用”的项：

| 设置 | 症状 | 处理 |
| --- | --- | --- |
| `write_are_deterministic_algorithms_enabled` | 默认开，声称把“本次运行要求确定性算术”写进启动记录；而 `filter_reduction_configs_for_determinism` 读的正是这个键，却没有任何生产者写入，于是即使要求了确定性，归约仍可能选到自由排序的配置 | 在 `tp_meta_common` 中按设置写入 `are_deterministic_algorithms_enabled` |
| `use_launch_metadata_schema` | 默认开，声称按“说明字段含义的表”读启动参数，而不是按编译对象身上恰好可读的字段；启动器仍在 `hasattr` 探测 `num_warps` / `shared` | 表存在且设置为开时，`num_warps` / `shared_mem` 取自 `binary.launch_metadata_schema`；无表或表缺该字段时仍按原路径探测 |
| `combo_kernel_allow_mixed_sizes` | 常量 `1`，无读者 | 保留：上游同样只定义不读，且本仓库的组合内核分组根本不按尺寸分组（按设备/流/内存池切块），凭空加“禁止混合尺寸”会是自行发明的语义 |

顺带说明 `is_fbcode()`：本仓库它是恒定 `False`，因此快照/缓存键里不写它不会带来跨构建差异；
上游把它写进自动调优缓存键是为区分内部与外部构建，本实现没有这种区分。

### 8.6 部署包：编译产物的运行时形态

`compiler.build_aot_package(fn, example_inputs, path)` 把一个区域写成一个目录：

| 文件 | 内容 |
| --- | --- |
| `model.so` | 生成的 host kernel 与入口 `aoti_entry(const void* const* inputs, void* const* outputs, const long* sizevars)` |
| `weights.bin` | 程序从周围读到的常量，按槽位顺序打包；编译期已转成连续布局 |
| `metadata.json` | 槽位表：输入 / 输出 / 常量 / scratch，含形状、步长、dtype、设备与常量在权重文件中的偏移 |
| `aoti_metadata.h` | 同一张表的 C++ `constexpr` 形式，C++ 调用方据此知道参数形状而不用解析 JSON |
| `aoti_entry.cpp` | 按上表生成的 C 入口，`aoti_set_constants(const void* const*)` 交出常量 |

要点与边界：

- 入口收的是**裸缓冲区**：输入按区域参数顺序，输出由调用方提供，因此同一个包可以反复调用而结果互不影响。
- 包只链接运行期库（`libp10`），不需要 Python；测试里用一个不 import 编译器的子进程与一个 `g++`
  编译的程序分别跑通同一条入口。
- 只覆盖 C++ wrapper 能写出的 CPU 调度（逐点链、`mm`/`matmul`/`bmm`/`linear`/`addmm`/`conv2d`、softmax、layer_norm 等；返回视图的区域拒绝）：
  Triton kernel（如本机 CPU 上的卷积）与回退到图执行器的区域在构建时即拒绝。
- 构建以 `-Wl,-Bsymbolic` 链接、生成头里的常量用内部链接：同进程加载多个包时，各自的槽位表与
  入口互不覆盖（否则链接器会把 `STB_GNU_UNIQUE` 的表统一成同一个）。
- 未做：GPU 包（需要把 Triton kernel 一并打包）、符号尺寸包（`sizevars` 目前只记录不填值）、
  把包接进 `compile()` 的一等公民选项。

### 8.7 host wrapper：哪些区域写得出来，以及三处错读

`stax.cuda_codegen` 把一个区域打印成一个 C++ 翻译单元，导出 `call(TpTensorBuffer*, long*)`。
此前这条路径没有任何测试，覆盖情况与三处缺陷如下。

**写不出来的区域**：kernel 由 Triton 生成、回退到图执行器、或调用没有 shim 的外部 kernel
（`sort`/`cumsum` 之类），会在降级时明确报错，而不是写出一个算不出结果的包。
`_op_name_from_kernel` 现在把 kernel 名尾部的重载后缀（`_out` / `_default`）也去掉，因此
`matmul.default` 这类"默认重载"能对上已注册的 `matmul`。

**shim 的覆盖面**：`linear`（偏置可有可无）、`addmm`（带 `beta`/`alpha` 两个数）与
`convolution`（带 `stride`/`padding`/`dilation`/`transposed`/`output_padding`/`groups`，偏置同样
可有可无）各有自己的 shim。参数分两类：张量走槽位，其余按"字面量"写进调用——整数仍写成整数、
实数仍写成实数、列表写成 `std::vector<int64_t>{...}`、标志写成 `true`/`false`、缺省写成
`std::nullopt`，因为参数的类型与形态也是这个运算含义的一部分。同一个运算可以用不同的字面量调用
多次（padding=1 与 padding=0），每次调用各自生成一个 shim，名字按出现顺序加后缀区分。

**三处缺陷**：

1. `ExternKernel.set_cpp_kernel_name` 把 `op_overload.name` 当字段读。它是一个方法，取到的是绑定
   方法对象，随后 `.replace(...)` 抛 `AttributeError`。结果：`a @ w` 这类节点在 C++ wrapper 下
   直接降级（包也就写不出来）。现在按"可调用就调用"处理，与 `_CollectiveKernel` 里同一写法一致。
2. 更严重的一处是**读错缓冲区**：调度器允许把"读一个缓冲、写一个缓冲"的逐点 kernel 判成原地
   （`make_inplace(input, output)`），kernel 体因此只收一个 `in_out_ptr0`；但 wrapper 给每个名字
   分配的是各自独立的槽位，于是传给 kernel 的是**结果**缓冲，输入缓冲（上一个 kernel 的输出）
   从未被读。表现为 `mm(x, W).tanh()` 之类返回未初始化内存的值——只有 `relu`/`abs` 这类幂等逐点
   恰好"看起来对"。`CppWrapperCodegen.can_reuse` 现在返回 `False`：这条路径没有 `codegen_inplace_reuse`
   的对应实现，与其复用存储不如多给一个缓冲。

3. **视图（reshape / transpose / 切片）读不到存储**。C++ wrapper 给每个名字分配一个槽位，
   而视图没有自己的存储：它由 `codegen_reinterpret_view` 生成形如
   `reinterpret_tensor(buf, (3, 2), (1, 3), 0)` 的表达式来命名，这个名字既不是缓冲区名，
   `buffer_meta` 里也没有它。后果有两层——作为输出时 `call` 报
   `output buffer ... was not allocated`；作为 kernel 参数时槽位 filled 的是全零
   （`data` 为空指针），kernel 于是从未读到真正的数据。现在 `codegen_reinterpret_view` 会把
   "名字 → 底层缓冲 + 形状 + 步长 + 偏移 + 类型"记进 `view_meta`，`call` 先分配底层缓冲、
   再逐个把视图槽位指向"底层地址 + 偏移×元素大小"，返回值时用 `as_strided` 还原视图。
   `CppWrapperModule` 因此能返回视图，`aoti_package` 则明确拒绝含视图的区域（包要自己排布存储）。
   顺带修掉一处**变窄**：`_generate_extern_kernel_alloc_helper` 原先用 `input.get_name()` 取参数名，
   视图会因此退回到底层缓冲的名字，把整块缓冲而不是窗口交给 kernel；现在统一走
   `codegen_reference()`，与 Python wrapper 的 `val_to_arg_str` 一致。

**另外两处（不在 wrapper，而在下降路径）**：

1. `InputsKernel.unwrap_storage_for_input` 只把 `BaseView` 走一遍 `realize_input`，其它 IR 节点
   直接 `raise AssertionError`。于是任何"先把值算出来再交给矩阵乘"的写法——`mm(relu(x), W)`、
   `mm(x + b, W)`、`mm(cat([x, x]), W)`——都会在下降期被拒（CPU/CUDA 两条路一样）。现在非
   `Buffer`/`ReinterpretView` 的 IR 节点也会被 `realize_input` 落成缓冲，调用读到的是那份拷贝，
   产出它的 kernel 排在其前面。
2. `convolution` 的下降对 `stride`/`padding`/`dilation`/`output_padding` 直接 `tuple(...)`。调用是
   按写的样子记录的，而前端允许用一个数表示每个轴（`padding=1`），于是从 `Tracer` 出来的图里这些
   参数是**整数**，下降期 `TypeError: 'int' object is not iterable`——部署包路径正是这样拿到图的，
   所以 `build_package` 对带 padding 的卷积直接失败。现在下降先把单个数补成单元素序列（与前端
   `tensorplay.functional.conv2d` 的归一一致），前反向两条下降都走同一处理。

这两处修完之后，`conv2d`（含 bias、padding、stride）与"卷积 → 逐点 → 卷积"的网络不仅能编译，
还能写成部署包：§8.6 的包覆盖因此从全连接网络扩到卷积网络。

`test/stax/test_stax_cpp_wrapper.py`（34 项）覆盖逐点链、多输出、常量、`mm`/`matmul`/`bmm`/
`linear`（带与不带偏置）、`addmm`（含"矩阵乘后加偏置"）、`conv2d`（无偏置/带偏置/padding 与
stride/后接逐点/两层卷积网络）、两层全连接网络，以及"外部 kernel 之后接逐点"的整族
（`tanh`/`exp`/`neg`/乘加/`abs`/`relu`、逐点链、归约、linear 接逐点）；另有 8 项覆盖视图：
转置/reshape/切片结果、切片偏移、转置输入、同一缓冲的两个视图、输入自身的视图。每个用例都断言
**没有发生降级**，因为降级路径同样能算出正确结果。下降路径那处修复由
`test/stax/test_stax_native_lowerings.py::test_a_call_reads_a_value_that_is_still_being_computed`
覆盖（同样断言没有降级）。

## 9. 算子与分解覆盖

- `operator_coverage.py` 设计是好的（显式声明 LOWERED/TEMPLATE/LIBRARY_CALL/NOT_COVERED，并有测试），但：
  - 表已过期：`index_add`、`index_copy`、`scatter_add` 在 `NOT_COVERED` 中，实际已有 lowering（`op_lowerings.py:6025,6048,6101`）；
  - 文档说 `NOT_COVERED` 会抛，实际是 extern fallback；
  - 没有调用者，纯文档性。
- `LOWERINGS` 实际约 850 个算子名，`_TEMPLATE_OPERATORS` 8 个（conv/matmul/linear/addmv 等）。
- `tensorplay/_decomp` 注册约 480 个分解，上游 `torch/_decomp` 约 180 个 + inductor 专属表；本仓库在分解数量上是超集，但缺少 inductor 侧的一些 lowering/pattern 组合。
- ~~`fft`、`linalg_*`~~：复核为已有。`_decomp/decompositions.py` 注册了 20 个 fft 分解（一维 `fft`/`ifft`/`rfft`/`irfft`、n 维与 2 维族、`fftshift` 族），落在 `prims.fft_r2c/c2c/c2r` 原语上（prims 已有 `_tp_functional` 实现）；`linalg.vector_norm`/`inv`/`svd` 编译结果正确且图中无 fallback。
- ~~`embedding_bag` 的缺口在算子层~~：复核为**注册缺口**而非 kernel 缺失。CPU kernel（`EmbeddingBagKernels.cpp`）一直存在，但只注册了 `_embedding_bag*` 内部拼写，公开 overload `embedding_bag` / `embedding_bag.padding_idx` 无注册，eager 直接报 `Kernel not found`。已修复：两个公开 overload 注册为委托同一前向的包装。移植时注意到生成签名（`KernelSignatures.h`）里 `int?` 参数**按值**传 `std::optional<int64_t>`，wrapper 写成 `const&` 会 ABI 不匹配段错误——这正是段错误的来源。回归测试 `test/nn/test_embedding_bag_cpu.py`（4 项：默认/padding/mean 对照手写参考、编译路径与 eager 对拍）。
- `histogram`、`unique_dim`、`segment_reduce`、`index_reduce`、稀疏 COO/CSR、jagged 的原生 lowering，主要靠 library call/fallback。其中 histogram / unique 族 / segment_reduce / index_reduce / scatter_reduce_ 是**刻意**的 fallback（`op_lowerings.py` 尾部逐一注明理由：答案的形状就是答案本身，无法按已知长度的范围写出遍历；散布归约的胜者是语义选择），不是遗漏。

---

## 10. 优先级建议

### P0（先修正确性）

1. ~~`graph_artifact_cache` 键纳入所有输入（含标量/容器），提升 schema 版本；回归测试。~~ 已完成。
2. ~~数据相关输出尺寸读取：明确抛错，回到“不支持就拒绝”。~~ 已完成。
3. ~~决定 `graph_passes` 去留：post-grad 与联合图都已接线。~~ 已完成。
4. ~~选项生效或明确拒绝：`stax.fusion` / `stax.triton` / `stax.max_autotune`。~~ 已完成。

### P1（能力对齐）

1. ~~自动 dynamic + `dynamic_shapes=Dim` + `mark_static` / `disallow_in_graph` 真正生效。~~
   已完成（§3.2、§13）：自动 dynamic 只在“仅尺寸变化”的守卫失配上触发；`dynamic_shapes` 支持映射/位置/装饰器三种写法与 `Dim` 约束；`mark_static` 在编译前与捕获中都生效；`disallow_in_graph` 整段拒绝而不再静默吸收。
2. ~~补 §4.3 里仍缺的 pass（`reorder_for_locality`、`remove_assert_ops`、`split_cat` 接线等）。~~ 已完成（`remove_profiler_ops` 除外，本仓库无对应算子）。
3. ~~CPU GEMM 模板 + micro-gemm 变体~~ 已完成（§6、§13）：`CppGemmTemplate` + AVX2/AVX512 micro-gemm
   （f32/bf16/f16），预打包权重、int8/woq/AMX/Brgemm 与动态 M 明确不做；oneDNN/MKLDNN、quantized lowering 仍缺。
4. AOTInductor 部署包；cudagraph trees。
5. 设备 codegen（MPS/XPU/ROCm）或明确拒绝。
6. ~~整图 cache~~ 已完成（§8.1、§13）；分布式 collective lowering/overlap 仍缺。

### P2（清理与一致性）

1. ~~修死 import；删除不可达的移植残留。~~ 已完成。
2. ~~配置面与文档数字对齐。~~ 已完成（§8.5）：补齐 4 个被读取却未绑定的设置并新增 `save_config_portable`，
   修掉两处读错命名空间；配置面数字更新为实测 195 项 / 1,894 行。
3. ~~补 `compile_worker`、pass、pattern matcher、foreach、split scan 等测试。~~ 已完成（§8.3）。

---

## 11. 复现脚本

### 11.1 标量参数陈旧

```bash
python - <<'PY'
import tensorplay as tp

def f(x, s):
    return x * s

c = tp.compile(f, backend="stax", dynamic=True)
x = tp.randn(3, 4)
print(tp.equal(c(x, 2), x * 2))  # True
print(tp.equal(c(x, 3), x * 3))  # False（bug）
PY
```

关闭工件缓存可验证根因：

```bash
python - <<'PY'
import tensorplay as tp
from tensorplay.compiler.backends.stax import config as stax_config
stax_config.force_disable_caches = True

def f(x, s):
    return x * s

c = tp.compile(f, backend="stax", dynamic=True)
x = tp.randn(3, 4)
print(tp.equal(c(x, 2), x * 2), tp.equal(c(x, 3), x * 3))  # True True
PY
```

### 11.2 数据相关尺寸

修复前：第二次调用返回 `[0,0,0,0]`（期望 `[-2,-2,-2,-2]`）。

修复后：`y.numel()` 在捕获期直接抛 `GraphCaptureError`；`strict_native=True` 时降级阶段也会以 `NotImplementedError` 拒绝。只有“把选中的值当结果返回”的写法照常编译。

```bash
python - <<'PY'
import tensorplay as tp

def f(x):
    y = x[x > 0]
    return x + 1 if y.numel() > 0 else x - 1

try:
    c = tp.compile(f, backend="stax")
    c(tp.ones(4))
except Exception as exc:      # 修复后：捕获期就拒绝
    print(type(exc).__name__, exc)
PY
```

### 11.3 副作用

```bash
python - <<'PY'
import tensorplay as tp
calls = []

def f(x):
    calls.append(1)
    return x + 1

c = tp.compile(f, fullgraph=True, strict_native=True)
for _ in range(3):
    c(tp.randn(2, 2))
print(len(calls))  # 1（上游语义下应为 3）
PY
```

### 11.4 测试

```bash
pytest test/stax -q -p no:randomly
```

---

## 12. 审计边界与不确定项

- 未做的事：全量上游测试对拍、真实模型端到端 benchmark、CUDA/HIP/XPU 多设备实测、AOTI 产物验证、分布式多机验证。
- 静态推断但未运行验证的项：
  - `graph_artifact_cache` 键遗漏对**张量以外**输入的影响范围（已用标量复现，容器/对象未逐一验证）；
  - 分布式 collective IR 是否在未审计路径被构造（主路径未发现构造点）；
  - 缓存键对工具链版本不敏感导致的跨环境复用风险；
  - GEMM 决策缓存未校验候选表导致的错误复用。
- 行号基于本次审计时的 working tree；仓库有多方未提交改动，后续可能漂移。

---

## 13. 本轮修复记录

| 提交 | 内容 |
| --- | --- |
| `8e309826` `fix(compiler)` | 工件缓存的键覆盖全部样例输入（含标量/容器），schema 升到 `2` |
| `925fd78d` `test(compiler)` | 补标量区域输入的缓存回归用例；修一个 dtype 容差导致的偶发失败 |
| `4f4a71ff` `fix(compiler)` | 数据决定长度：捕获期拒绝读尺寸，降级期拒绝按样例定尺寸，其余情形回退 |
| `94cb88d8` `refactor(compiler)` | 清理 12 处指向不存在模块 / 点层数错误的 import，改为明确拒绝 |
| `43ea8eac` `fix(compiler)` | 去掉覆盖环境变量读取的重复配置定义 |
| `6f0dd82b` `feat(compiler)` | 把 post-grad 图重写接进主路径；选项转为一次编译内的配置上下文 |
| 本轮后续 `feat(compiler)` | 联合图 pass 接进 AOT 分区前；量化 canonicalize 缺席即跳过；`GraphTransformObserver` 不再无条件 `to_dot()` |
| `2953ff64` `fix(compiler)` | 常量折叠器改为真正的图遍历，切分用的 import 修正 |
| `aa9d47b6` `fix(compiler)` | 让联合图重写在“本仓库记录的值”上跑得通（见 §13.1） |
| `f32234fb` `fix(compiler)` | 模式匹配按参数名绑定调用，关键字写法的模式终于能命中位置参数记录；`split_cat` 认 `split.sizes` |
| `790d42b8` `feat(compiler)` | post-grad 补 DCE、局部性重排（随机算子与写回边界不跨）、删除元数据断言、`1/sqrt → rsqrt`、按名启用 `split_cat` 表 |
| `aa8a674d` `test(compiler)` | 新增 `test/stax/test_stax_post_grad_passes.py`（7 项） |
| `4c81959b` `feat(compiler)` | profiler span 写进图（两种记录都认）并在 custom pre pass 之后删除；前端开关默认关闭；新增 `test/stax/test_stax_profiler_spans.py`（6 项，见 §4.6） |
| `dcf2b1d3` `feat(compiler)` | `pass_patterns[2]` 新增两条无条件重写：切分被全部按序拼回、`cat` 被沿接缝切回片段时整对删除；两种 split 记法都认；测试补 6 项 |
| `979f5757` `feat(compiler)` | `tp.compile` 统一到记录感知的 facade；捕获根按模块全局与闭包包重建；`disallow_in_graph` 登记并由捕获适配器和 `create_proxy` 双重拒绝；`mark_static` 记录/校验轴并转发捕获内标记 |
| `a828ff04` `feat(compiler)` | 新增 `_core/shape_policy.py`；`dynamic_shapes` 支持映射/位置/`Dim`/范围/静态 int；自动 dynamic（`config.automatic_dynamic_shapes`）；符号捕获按声明策略建符号并读钉住尺寸；导出 `check_dynamic_shapes` |
| `6240f4f1` `test(compiler)` | 新增 `test/stax/test_stax_dynamic_shapes.py`（18 项）；`sin(x)+1` 近零点改用与其余用例一致的绝对容差，消除一 ulp 差异造成的偶发失败 |
| `9566deaf` `fix(compiler)` | 捕获守卫补声明上下界与“每个引用同一符号的轴尺寸相等”；派生维样例尺寸必须可由关系产生并与根维的样例尺寸一致；`mark_static` 轴在样例校验前从声明策略中移除 |
| `393b01a5` `test(compiler)` | 补派生维不可达尺寸（首调与已捕获后）、根维/派生维样例不一致、声明上界在后续调用生效；导出侧补前两项 |
| `269cc3eb` `fix(compiler)` | 非张量全局/闭包/模块属性进入特化键：不可变标量与元组按精确内容、可变容器与可调用对象按身份；枚举范围扩到子模块与模块运行代码可达的方法；代码自身写回的名字不计 |
| `0f45c09b` `fix(compiler)` | `memory_budget` 真正约束留给反向的激活量（0 全重算、1 最省割、中间取可装入的候选割并按字节丢弃）；`keep_inference_input_mutations` 真正选择更新在图内还是运行后拷回；移出更新的图模块记录被写参数位置 |
| `8f15f15c` `test(compiler)` | 预算按字节生效与越界拒绝的用例；两种输入更新位置的结果与图形态；无更新时两种取值不变 |
| `3b8a9207` `fix(compiler)` | 补齐移植模块读取却未绑定的 4 个设置；新增 `save_config_portable`；bucketing 改读存在的分布式命名空间；CUDA 精度上下文回退到进程级精度并把未暴露的缩减开关按缺席处理 |
| `2431f64a` `test(compiler)` | 新增 `test/stax/test_stax_config_surface.py`（12 项）：读取名存在、编译线程首次使用时决定、带宽覆盖、bucket 模式来源、快照可写文本/分节/随设置变化/过滤私有名与自身辅助函数、隔离键稳定且随选择收缩 |
| `c9d7abae` `test(compiler)` | 新增 `test/stax/test_stax_compile_worker.py`（8 项）与 `test/stax/test_stax_pattern_foreach_scan.py`（11 项），见 §8.3 |
| `f9224c6f` `fix(compiler)` | `config_comms` 补上被读却未绑定的 `sink_waits_iterative_swap_with_collectives`；测试遍历后端所有 `config_comms.` 引用确认可解析 |
| `3758e5fc` `fix(compiler)` | 覆盖表删除 6 个已有 lowering 的 `NOT_COVERED` 条目；分解需要的名字若已有 lowering/模板则不再折进未覆盖集；测试加断言禁止声明与实表漂移 |
| `56320e21` `fix(compiler)` | 反向规则改为有序贡献表，按使用位置累加；同一节点填两个操作数（平方/立方/自加/自减/方法拼写/两路汇合/派生值复用）的梯度在两种分区下均补齐 |
| `0747cc7d` `fix(compiler)` | 部分归约的梯度先补单例再回铺；单轴/负轴/命名轴/双轴/两种 keepdim 在两种分区下与按需重算一致 |
| `8b41d4e5` `fix(compiler)` | 反向占位符不对应参数时从模块属性读取被提升的常量；区域自建常量的梯度用例覆盖两种分区与两种保存策略 |
| `a338812c` `fix(compiler)` | 新增 `check_backend_kwargs`；eager/onnxrt/cudagraphs/未绑定选项的 AOT 后端拒绝不能兑现的 `mode`/`options`；cudagraphs 后端按 `dynamic` 跳过并在 `strict_native` 下报错；tvm 把区域名与 `dynamic` 当描述而非选项；测试补 4 项 |
| `0d94a967` `fix(compiler)` | 多输出区域的联合输出压平并把前向结果个数传给两种分区器；两结果区域在两种分区与两种保存策略下与 eager 一致 |
| `f6f8f2e8` `fix(compiler)` | 方法拼写的原地写算作写并留在前向；写前有读者时做前置副本，只有写后读者的直接用写后值；写前写后都有读者、以及写进计算值的情形明确拒绝；四种组合的梯度与缓冲结果对齐 |
| `479e3158` `fix(compiler)` | 前向闭包保留所有确有效果的调用（不止写值）；训练区域在重写与切分之间不再丢掉 profiler span |
| `0c49b8aa` `fix(compiler)` | 移除 `isolate_recompiles`：它被传递、被存下、无人读取；本前端本就按区域独立保存特化，跨调用共享条目不存在，两种取值都明确拒绝 |
| `413894de` `fix(compiler)` | `name` 从“传给后端即被丢弃”变为真正的调试元数据：调用对象上保留、用于重编译警告与两条上限错误的区域名、校验为字符串、不进入任何键 |
| `7f69aa3c` `fix(compiler)` | 确定性算术开关写进启动记录（归约配置过滤读的就是这个键）；`use_launch_metadata_schema` 让 `num_warps`/`shared_mem` 取自二进制自述表，缺表或缺字段时回退探测；新增 `test/stax/test_stax_launch_metadata.py`（5 项） |
| `b9d3a36b` `fix(frontend)` | `export(strict=...)` 不再接受两种取值后丢弃：只有“记录一次执行”这一条捕获，`strict=False` 是它，`strict=True` 明确拒绝并说明没有第二条路；非 bool 报 `TypeError` |
| `c94f9159` `fix(compiler)` | 区域名不转发给后端（前端自己写在调用对象与诊断里），`check_backend_kwargs` 不再把 `name` 当区域关键字；Stax 入口删掉被丢弃的 `name` 参数与 `_lower_stax_region` 上四个不读的参数；相关用例改为断言新契约 |
| `1e4ad231` `docs(compiler)` | 改写 `export` / `unused` / `overload_method` 的文档为“脚本化侧标记、不改变捕获”；`substitute_in_graph` 的两个标记不再存进记录，折叠提示写明“捕获本就如此”；新增 `test/stax/test_stax_annotations.py`（6 项） |
| `1c55d7e5` `docs(compiler)` | 审计报告补“注解只影响脚本化侧、`substitute_in_graph` 的两个标记不再入库” |
| `77bd3888` `docs(compiler)` | 审计报告补 `export(strict=...)` 的拒绝（§8.4、§0、§13） |
| `d92d739a` `feat(compiler)` | 整图缓存：`graph_artifact_cache` 重写为区域级工件（键含程序、样例输入、模块状态、全部设置、包版本、编译器与系统指纹），命中即跳过降级/调度/打印/构建；符号形状不入缓存；新增 `test/stax/test_stax_graph_cache.py`（7 项） |
| `92e5785f` `fix(compiler)` | `ComputedBuffer` 的读名/读数直接问 body，不再为了知道读什么而执行 store |
| `181cd820` `fix(compiler)` | 外部调用不再被要求自己定布局：只填缓冲的调用对元素顺序无所谓，有要求的调用自己覆写 |
| `8b844fa6` `fix(compiler)` | 尺寸提示覆盖所有剩余自由符号（含 kernel 自造的循环符号），除零时回退而不是报错 |
| `d880ca6d` `fix(compiler)` | 外部 kernel 以“能接目标缓冲”的形式（操作包）注册；被测量的是程序会写的那次调用 |
| `a148dfee` `fix(compiler)` | 交给外部调用的值先 realize 再查 stride-1，未落地的值不再被判为无 stride |
| `a35c7bea` `feat(compiler)` | CPU 分块乘积模板：`CppGemmTemplate` + AVX2/AVX512 micro-gemm（f32/bf16/f16）、线程/缓存分块、偏置与可折叠 epilogue 融进 store；`mm`/`addmm` 的 `.default` 拼写一并指向调优 lowering；框架 kernel 作为候选需两条设置一致；新增 `test/stax/test_stax_cpu_gemm_template.py`（13 项） |
| `3966d1ed` `fix(compiler)` | 结果的布局用“写出来的尺寸”描述：`int()` 曾经在符号尺寸上直接抛错，而形状变化的第二个调用就会按符号捕获 |
| `3961b439` `fix(compiler)` | 带偏置的乘积接受函数写法记录的位置参数（先加数系数、后乘积系数，见 §13.1） |
| `a6a6951e` `docs(compiler)` | 审计报告记录整图缓存与 CPU 分块乘积，并把刻意不做的部分（预打包、整数路径、运行期才知道的行数）写在旁边 |
| 本轮 `fix(compiler)` | 全局/闭包/模块属性中的可变容器按内容快照进入特化键，原地编辑（`list.append`、`dict[k]=v`）触发重捕；捕获代码自己原地修改的名字经字节码扫描排除出键（见 §3.3）。别名输入（同 storage）且图内含原地写时，捕获期以 `GraphCaptureError` 明确拒绝；只读别名与独立输入的原地写照常编译。新增用例：`test_stax_captured_values.py` 2 项、`test_guards.py` 3 项 |
| 本轮 `feat(compiler)` | post-grad 补 `view_to_reshape`（`view.default` 在产物里统一为 `reshape.default`）与 `unfuse_bias_add` / `unfuse_bias_baddbmm`（bias 类加数 + 全逐点使用者的 GPU 区域改写成逐点链；判定设备读自区域真实设备，模式替身报的是 meta 设备）。新增用例：`test_stax_post_grad_passes.py` 3 项（view 改名、CUDA 解融、非 bias 加数保持融合）；§4.3 的 `cat_slice_cat` 条目复核为已有，记录更正 |
| 本轮 `docs(compiler)` | §9 复核更正：fft（20 个分解落 `prims.fft_r2c/c2c/c2r`）与 `linalg.vector_norm`/`inv`/`svd` 编译正确且无 fallback，原条目过时；histogram / unique 族 / segment_reduce / index_reduce / scatter_reduce_ 确认为刻意 fallback 并注明理由 |
| 本轮 `fix(kernels)` | `embedding_bag` 公开 overload 注册到 CPU（用户指出算子存在后复核发现）：kernel 一直在，缺的是 `embedding_bag` / `embedding_bag.padding_idx` 的 `m.impl` 注册；`int?` 参数按生成签名以值传 `std::optional<int64_t>`，写成 `const&` 会 ABI 段错误。新增 `test/nn/test_embedding_bag_cpu.py`（4 项） |
| 本轮 `feat(compiler)` | `pointless_cumsum` 落地：`cumsum(full(v), dim)` 的扫描改写为常量 ramp 的节点链（arange→mul→view→expand→to），fill 有其他读者时保留 fill；标量 `cumsum` 不归此 pass。上游的 `FakeTensorUpdater.incremental_update()` 接到 pass 末尾在本构建上无法验证为无害（rms_norm 的 row kernel 选择丢失，但二分证明该失败在会话前的 pass 版本上同样出现，来源是进行中的 p10 内核改动与构建漂移，不是刷新本身）；本仓库降级前的 `ShapeProp` 以真设备模式给新造节点填假值，`pointless_cumsum` 不依赖全图刷新。新增用例：`test_stax_post_grad_passes.py` 3 项（浮点、整数、程序数据的 cumsum 保留） |
| `13664739` `feat(compiler)` | post-grad 新增加数与乘积、缩放商与加和的合成：`add(mm(a, b), x)` / `add(x, mm(a, b))` 合成一次带偏置乘积（加数可广播、三方同型、`alpha == 1`，且不抢 `unfuse_bias_add` 的 GPU 逐点链）；分解成 `div → mul → add` 的 `addcdiv` 在 CUDA/XPU 上合成一次带缩放商调用（商只舍入一次、乘积与加和走 FMA；带舍入模式或张量缩放不合成）。新增用例 6 项 |
| `25da76a7` `test(compiler)` | `test/stax` 会话级隔离 `TP_CACHE_DIR` 并默认关闭整图工件缓存，`test_stax_graph_cache.py` 自行开启：观察降级过程的用例不再因同会话前一个用例写入的工件命中而误报 |
| `5a0dc3e4` `fix(compiler)` | `upsample_nearest{1,2,3}d` 接受与输出尺寸一致的缩放提示（此前一律视为边界，`F.interpolate(scale_factor=…)` 区域因此回退到图执行器）；缩放与尺寸矛盾时仍回退，符号尺寸无法核对时按尺寸比例映射 |
| `72503adf` `fix(compiler)` | `one_hot` 补分解为 `unsqueeze → eq(arange) → to(int64)`，范围检查作为不参与死代码删除的调用保留；`_assert_async.msg` 的 lowering 改为接受 `(cond, msg)`；分解注册表补 `one_hot.default` 样例 |
| CUDA Graph Trees | 新增 `backends/cudagraph_trees.py` 与 `triton.cudagraph_trees`（默认开）：按输入布局分节点记录在一条树上，节点共享一个分配器池；录制持有私有输入副本，回放经 `stage_and_launch`（无该入口时逐张量 `copy_`）写入后启动；输出克隆返回，调用者结果跨回放有效；上限按最近最少使用淘汰。`CudagraphCompiledCallable` 按布局派生节点键，布局变化录制子节点而不是回退到未捕获工件；树关时保留原直连路径。新增 `test/stax/test_cudagraph_trees.py`（17 项），改写的 wrapper 用例覆盖换地址回放、输出独立与跨布局录制（§8.2） |
| host wrapper 的正确性与覆盖面 | `_op_name_from_kernel` 去掉尾部的 `_out` / `_default` 重载后缀，`matmul.default` 能对上已注册的 `matmul`；`ExternKernel.set_cpp_kernel_name` 把 `name` 当方法调用（此前当成字段，取到绑定方法后 `.replace` 抛 `AttributeError`，`a @ w` 因此在 C++ wrapper 下静默降级）；新增 `linear` 与 `addmm` 的 shim（前者按张量个数分表，后者把 `beta`/`alpha` 按原样写进调用）；`CppWrapperCodegen.can_reuse` 返回 `False`，因为该路径没有 `codegen_inplace_reuse` 的对应实现——此前调度器把“读一个缓冲写一个缓冲”的逐点 kernel 判成原地，wrapper 却把**结果**缓冲传给 kernel，输入缓冲从未被读（`mm(x,W).tanh()` 之类返回未初始化内存，只有幂等逐点偶然正确）；视图（`transpose`/`reshape`/切片）此前没有槽位也没有元数据：作为输出时 `call` 报 `was not allocated`，作为参数时读到空指针；现在 `codegen_reinterpret_view` 记下“名字 → 底层缓冲 + 形状/步长/偏移/类型”，`call` 逐层解析后把视图槽位指到底层地址 + 偏移，输出用 `as_strided` 还原，包侧则明确拒绝含视图的区域；新增 `test/stax/test_stax_cpp_wrapper.py`（29 项，该路径此前无任何测试，§8.7） |
| 卷积：数值参数与字面量 shim | `convolution` 的下降对 `stride`/`padding`/`dilation`/`output_padding` 直接 `tuple(...)`，而调用是按写的样子记录的（`padding=1` 记成整数），于是从 `Tracer` 出来的图在下降期 `TypeError`，部署包路径因此无法打包带 padding 的卷积；现在下降先把单个数补成单元素序列，前反向两条下降一致。C++ wrapper 侧新增 `convolution` 的 shim，参数分“张量”与“字面量”两类（整数/实数/列表/标志/缺省），同一运算的不同字面量各自生成一个 shim 并以名字后缀区分（§8.7） |
| 下降路径：调用读到仍在计算的值 | `InputsKernel.unwrap_storage_for_input` 此前只对 `BaseView` 走 `realize_input`，其它 IR 节点直接 `raise AssertionError`，于是 `mm(relu(x), W)`、`mm(x + b, W)`、`mm(cat([x, x]), W)` 这类写法在下降期被拒（CPU/CUDA 两条路一样）；现在非 `Buffer`/`ReinterpretView` 的 IR 节点也会被落成缓冲，调用读那份拷贝，产出它的 kernel 排在其前（§8.7） |
| AOT 部署包 | 新增 `backends/stax/aot_package.py` 与 `compiler.build_aot_package` / `compiler.load_aot_package`：区域写成 `model.so` + `weights.bin` + `metadata.json` + `aoti_metadata.h` + `aoti_entry.cpp`，入口 `aoti_entry(inputs, outputs, sizevars)`；常量在编译期转成连续缓冲后烘进权重文件，元数据同时给出 JSON 与 C++ `constexpr` 两种形式；包只依赖运行期库，不需要 Python。构建脚本与 `_build_cpp_wrapper` 共用 `cpp_wrapper_build_options()`；包以 `-Wl,-Bsymbolic` 链接、头文件常量用内部链接，同一进程加载多个包不会互相串符号。新增 `test/stax/test_stax_aot_package.py`（18 项，含两层网络、无 Python 的 `ctypes` 子进程与编译出的 C++ 程序两条运行路径） |

修复验证以 `pytest test/stax` 全量一次跑完为准，结果见 §0。

### 13.1 接线后才暴露的潜在缺陷

把从不运行的 pass 接上主路径，等于第一次执行它们；下面这些缺陷都是这样露出来的，**它们与本次接线无关，但会让接线后的编译直接错或直接崩**：

| 位置 | 症状 | 根因 | 处理 |
| --- | --- | --- | --- |
| `op_lowerings.py` | `TypeError: unexpected keyword argument 'allow_cpu_inputs'` | 同名函数被定义两次，后一个（不带该参数的旧版）胜出，`pattern_matcher` 调用的是它 | 删掉旧版，保留带参数的版本 |
| `constant_folding.py` | `TypeError: object.__init__() takes exactly one argument` | `class ConstantFolder(Any)`——上游是继承图解释器，移植时基类写成了 `Any` | 改为继承 `Interpreter` |
| 同上 | `TypeError: typing.Any cannot be used with isinstance()` | 4 处 `isinstance(x, Any)`（上游是 `isinstance(x, fx.Node)`），以及 `tree_map_only_(Any, ...)` | 换成 `Node` / `tp.Tensor` |
| `joint_graph.py` | `AttributeError: 'CustomOpDef' object has no attribute 'default'` | 本仓库 `prims.*` 是 `CustomOpDef`，直接用作 target，没有 `.default` | 按两种拼写比较 |
| 同上 | `AttributeError: _efficientzerotensor` | `tensorplay.ops` 对下划线开头的名字故意抛错，本构建也没有这个算子 | 缺席即跳过 |
| 同上 | 输出全变成 meta 张量 | 关节图的值只是“描述”（`_RecordedTensorMetadata`，设备常报 meta），而“把 meta 输出换成 `empty_strided`”这条规则把每一个输出都当成真的 meta 张量 | 只有区域本身跑在 meta 上时才做替换 |
| 同上 | `full: unexpected keyword argument 'layout'` | 本仓库 `full` 的 schema 没有 `layout` | 不再传 |
| `pad_mm.py` | `constant_pad_nd: missing required argument "value"` | 本仓库该算子 `value` 无默认值 | 显式传 `0` |
| 同上 | `AttributeError: fp32_precision` | 本构建的 cuBLAS 只有 `allow_tf32` | 先问新名字，没有再退回旧名字 |
| 同上 | `TypeError: randn_like() ... not _RecordedTensorMetadata` | “描述”不是张量，`is_fake_tensor` 不认它 | 描述一律按形状/步长/类型/设备造出真值再测 |
| `kernel_inputs.py` | `TypeError: Cannot convert symbols to int` | 结果布局从 `mnk()` 取尺寸，而 `mnk()` 把尺寸全部 `int()`；形状变化的第二次调用是按符号捕获的 | `mnk_symbolic()` 返回写出来的尺寸，`output_layout` 改用它；整数化只留给需要算分块/网格的调用者 |
| `templates/mm.py` | `TypeError: tuned_addmm() takes 3 positional arguments but 5 were given` | 函数写法 `tp.addmm(input, mat1, mat2, beta, alpha)` 把两个标量按位置记进图，而 lowering 只接受关键字；首次调用记录的是算子自身形态（标量是关键字），只在第二次捕获（形状变化）上暴露 | lowering 同时接受两种写法，位置顺序按函数声明的顺序 |
| `codecache.py` | `IsADirectoryError`（写本地缓存时） | 本地缓存文件与整图工件的目录用了同一个路径 `<cache>/cache/<system_hash>` | 本地缓存改为该目录下的 `local_cache.json` |
| `pattern_matcher.py` | `TypeError: 'NullHandler' object does not support the context manager protocol` | 仓库里有两个 `NullHandler`（`virtualized` 与 `ops_handler`），判断用的是另一个 | 按 `virtualized.NullHandler` 判断；模式替换在没有 fake mode 时用 `nullcontext` |
| `_core/api.py` | 各处 `AttributeError: is_cuda / is_contiguous / is_floating_point` | 值描述只保存了形状/类型/设备，pass 需要的判定没有 | 描述补上由形状/步长/设备/类型可直接推出的判定 |

另外 `post_grad_passes` 不再在每次改写后调用 `gm.recompile()`：编译产物读的是图，回退路径自己会重建，
这也是 `test/stax/test_compiler_backends.py` 明确要求的（成功时不得调用 `recompile`）。

结论：这批缺陷的共同模式是**“代码在、但从未执行”**——只要 pass 不接线，同名函数、错误的基类、
过期的算子签名、两个同名的哨兵类都不会被发现。因此“接线”本身也是这些缺陷的检查手段，
后续再补 pass 时建议同样先接一条最小路径并跑既有测试。
