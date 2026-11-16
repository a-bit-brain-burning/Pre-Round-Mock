# Day1 · 2026-11-16 模拟题 · csxsl 实现

在比赛平台选择 **NVIDIA H20** 和 **CUDA C++**。每道题将对应 `code.cpp` 的全部内容粘贴到编辑器，保留 `extern "C" void solution(...)` 入口。每份提交源码只依赖 CUDA Toolkit。

| 题号 | 题面 | 优化实现 | 默认配置 |
| --- | --- | --- | --- |
| 1 / A · All Pairs Shortest Path | [problem.md](1/problem.md) | [code.cpp](1/code.cpp) | 64 × 64 分块，FP32 |
| 2 / B · Argmax Over Dimension | [problem.md](2/problem.md) | [code.cpp](2/code.cpp) | 连续行使用 `float4`，跨步归约按输入形状分配线程 |
| 3 / C · Argmin Over Dimension | [problem.md](3/problem.md) | [code.cpp](3/code.cpp) | 连续行使用 `float4`，跨步归约按输入形状分配线程 |

`main` 分支保留相同路径的原始版本。用 `git diff main..csxsl -- Day1/1/code.cpp` 可比较第 1 题，其余两题同理。

## A 题

初始化时把对角线设为零，其他位置的零设为正无穷。在同一 CUDA stream 上按顺序更新对角块、同行和同列的方块、其余方块。一个线程块有 256 个线程。64 × 64 配置下，每个线程维护 16 个输出候选；结束时将不可达距离写为 `-1`。

文件顶部的 `FW_TILE` 可取 `32` 或 `64`，`FW_DPX` 可取 `0` 或 `1`。默认 `FW_TILE=64`、`FW_DPX=0`。比较分块时只调整 `FW_TILE`；比较整数 DPX 时保持分块大小相同，只改变 `FW_DPX`。

DPX 用于更新对角块所在行、列以外的方块。题面保证所有有限边权是正整数。每个线程块核对两个输入块的有限距离，均不超过 2²³ 时使用 `__viaddmin_s32`。此时两段距离相加不超过 2²⁴，FP32 可以精确表示。不可达距离使用整数标记 2²⁹，两个标记相加仍在 32 位有符号整数范围内。含有更大有限距离的块使用 FP32，原输出始终以 FP32 参与最后一次取最小值。范围检查与转换耗时包含在该配置的总时间内。

## B 题与 C 题

`input` 和 `output` 位于 GPU。入口使用 `cudaPointerGetAttributes` 确认 `shape` 的位置。对于普通 CPU 内存和固定页内存，入口在 CPU 上读取维度，按值传入归约长度、行数和步长。对于设备内存和 CUDA 托管内存，由 kernel 读取维度。有效输入通过一次 kernel 调用产生 `int` 下标，极值相同时选择首次出现的位置。

| 想比较的访问方式 | 调整文件顶部的值 | 受影响的公开输入 |
| --- | --- | --- |
| 每线程处理相邻两列，使用 `float2` 读取 | `ARG_COLUMNS_128` 从 `1` 改为 `2` | `(64,128,128,128), dim=2`，512 MiB |
| 每线程处理相邻两列，使用 `float2` 读取 | `ARG_COLUMNS_256` 从 `1` 改为 `2` | `(4,256,256,256), dim=1`，256 MiB |
| 一个 warp 处理一行 | `ARG_LONG_WARPS` 从 `4` 改为 `1` | `(8,1024,1024), dim=2`，32 MiB |

长度 64 的连续行由 16 个线程处理，每个线程读取一个 `float4`。线程组先按数值找到极值，再从取得该值的线程中选出最小下标。正零与负零同序，NaN 优先并保留最早下标。长度 1024 的连续行也按 `float4` 读取，默认由四个 warp 合并结果。输入首地址未满足 16 字节边界时，两种长度都使用标量读取。

对于 `(16,128,256), dim=1`，四个线程分别检查同一列中的四组行，再合并候选结果。对于 `(32,512,512), dim=0`，每个线程用 `float4` 连续读取四列，在寄存器中分别保留四个下标。另两个公开跨步用例默认由每个线程负责一列。CPU 已知维度时，入口根据工作量确定线程块数量，并为六个公开布局选择编译时已知长度和步长的 kernel。其他形状使用运行时维度。B、C 各自可以单文件提交，两份代码仅有 `FindMax` 的值不同。

### CPU 传入 `shape` 的计时

在 RTX 5060 Laptop GPU 上使用相同的 CPU 维度数组、GPU 输入和输出，交替测量支持 CPU 参数的提交 `5b2cf28` 与优化版本。六组用例的 CUDA event 耗时中位数之和如下。

| 算子 | `5b2cf28` | 优化版本 | 耗时减少 |
| --- | ---: | ---: | ---: |
| Argmax | 2.712 ms | 2.669 ms | 1.58% |
| Argmin | 2.796 ms | 2.754 ms | 1.50% |

长度 1024 的连续行改善最大。Argmax 从 56.53 µs 降至 29.26 µs，Argmin 从 51.93 µs 降至 25.74 µs。每个 warp 用整数归约指令获得极值的顺序值和首次出现的下标。[原始计时与源码 SHA-256](results-1116-reductions-20261116-081207.json)

### GPU 传入 `shape` 的历史计时

以下记录使用 GPU 上的 `shape`，在 RTX 5060 Laptop GPU 上比较 `90d676f` 与 `0a30066`。每组复用同一份输入，预热后测量 16 轮，轮流交换新旧版本的顺序。每轮用 CUDA event 测量连续 8 次调用并计算平均耗时。下表合计为六组用例各自的中位数之和。

| 算子 | 旧版合计 | 本次合计 | 耗时减少 |
| --- | ---: | ---: | ---: |
| Argmax | 2.975 ms | 2.713 ms | 8.8% |
| Argmin | 3.072 ms | 2.805 ms | 8.7% |

长度 64 的用例贡献最大，Argmax 从 0.621 ms 降至 0.380 ms，Argmin 从 0.623 ms 降至 0.380 ms。512 MiB 和 256 MiB 两组在此次测量中的变化均小于 0.3%。[各组原始计时与源码 SHA-256](results-1116-reductions-20261116-074928.json)

在仓库根目录运行 `flock --exclusive .cache/preround-gpu.lock python Day1/compare_reductions.py` 可重复 CPU 参数计时，按环境设置 `--nvcc`。测量 GPU 参数时使用 `--shape-device cuda`，可用 `--baseline` 指定 Git 基准版本。[对照脚本](compare_reductions.py)

## 验证与计时

[verify.py](verify.py) 需要 Linux/WSL、支持本机 GPU 和 SM90 的 nvcc、PyTorch、NumPy 和 SciPy。它为每种配置同时编译 SM90 和本机架构，通过平台的 C 函数入口调用源码。A 题用 SciPy 的有向 Dijkstra 核对全部输出，B、C 用 PyTorch 检查整数下标。测试覆盖所有公开尺寸，并单独检查跨块路径、不可达顶点、大整数、非整块尺寸、重复极值、NaN 和起始地址有一个元素位移的输入。

在本仓库根目录执行以下命令；本机需要已有上述依赖和可用的 NVIDIA GPU。

```bash
mkdir -p .cache
flock --exclusive .cache/preround-gpu.lock python Day1/verify.py
```

默认只检查正确性。需要重新计时时，显式加 `--timing-runs 5`：

```bash
flock --exclusive .cache/preround-gpu.lock python Day1/verify.py --timing-runs 5
```

若 nvcc 未加入 PATH，可用 `--nvcc /path/to/nvcc` 指定位置。编译缓存保存在已忽略的 `.cache/Day1/build/`。三个入口文件虽然使用 `.cpp` 后缀，验证脚本通过 `nvcc -x cu` 按 CUDA C++ 编译。新计时文件保存在 `Day1/`，源码 SHA-256 使用 `1/code.cpp`、`2/code.cpp`、`3/code.cpp` 路径作为键。

同一块物理 GPU 上若还同时运行原仓库或其他工作树的实验，必须让这些进程使用同一锁文件；可将上述 `.cache/preround-gpu.lock` 改为所有进程共用的绝对路径。

每次比较只改变一项配置，记录题号、宏的取值、正确性结果、GPU 原始耗时以及平台当天显示的基准。若平台显示分组耗时，A 题用 `n³ / 秒数` 计算每秒距离更新次数，B、C 用 `输入元素数 × 4 / 秒数` 计算有效读取带宽。后一个指标包括缓存与 kernel 执行效率的影响。

原仓库 11 月 16 日题面记录：A 的基准是 201.978 ms，四组总时间达到 134.652 ms 即得满分；B、C 各自的基准是 0.294887 ms，各自的满分时间约为 196.591 µs。每道归约题的六组输入合计 962 MiB。实际以平台当时显示的基准为准。

## 历史验证记录与来源

- 题面与原始版本来自 [Preround](https://github.com/a-bit-brain-burning/Preround) 的 `main@8b2795f7e846d3433b942fb88579a92d77462c56`，原路径为 `mock/11-16/{1,2,3}/`。`main` 中仅清理第 1 题代码的三处行末空白。
- 三个优化入口来自 `sim1116-csxsl@adc8a2bea4a01c9f0213569aa613b47d1d57b5e4`，源码按原始字节迁移，映射如下。

| 原路径 | 本仓库路径 |
| --- | --- |
| `sim-1116/floyd_warshall.cu` | [1/code.cpp](1/code.cpp) |
| `sim-1116/argmax.cu` | [2/code.cpp](2/code.cpp) |
| `sim-1116/argmin.cu` | [3/code.cpp](3/code.cpp) |

原仓库记录：2026-11-16 使用 CUDA 12.8.93 编译了全部 12 种配置的 SM90 和 SM120 代码。在 RTX 5060 Laptop GPU 上，A 的四种配置各通过 11 组检查，B 和 C 的四种配置各通过 15 组检查。A 的 `n=4096`，本机 FP32 的 32 × 32 与 64 × 64 配置耗时中位数分别为 69.84 ms、52.68 ms。**H20 的耗时与得分仍需由比赛平台测量。**

[原始五次计时及源码 SHA-256](results-1116-gpu-20261116-022947.json) 中的 `.cu` 文件名对应上表的旧路径。该记录的源码哈希对应源仓库 `1d0f196`，迁入时的源码来自 `adc8a2b`。
