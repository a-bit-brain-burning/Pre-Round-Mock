# Day1 · 2026-11-16 模拟题 · csxsl 实现

在比赛平台选择 **NVIDIA H20** 和 **CUDA C++**。每道题将对应 `code.cpp` 的全部内容粘贴到编辑器，保留 `extern "C" void solution(...)` 入口。每份提交源码只依赖 CUDA Toolkit。

| 题号 | 题面 | 优化实现 | 默认配置 |
| --- | --- | --- | --- |
| 1 / A · All Pairs Shortest Path | [problem.md](1/problem.md) | [code.cpp](1/code.cpp) | 64 × 64 分块，FP32 |
| 2 / B · Argmax Over Dimension | [problem.md](2/problem.md) | [code.cpp](2/code.cpp) | 跨步归约每线程读取一列，长度 1024 的连续行使用四个 warp |
| 3 / C · Argmin Over Dimension | [problem.md](3/problem.md) | [code.cpp](3/code.cpp) | 跨步归约每线程读取一列，长度 1024 的连续行使用四个 warp |

`main` 分支保留相同路径的原始版本。用 `git diff main..csxsl -- Day1/1/code.cpp` 可比较第 1 题，其余两题同理。

## A 题

初始化时把对角线设为零，其他位置的零设为正无穷。在同一 CUDA stream 上按顺序更新对角块、同行和同列的方块、其余方块。一个线程块有 256 个线程。64 × 64 配置下，每个线程维护 16 个输出候选；结束时将不可达距离写为 `-1`。

文件顶部的 `FW_TILE` 可取 `32` 或 `64`，`FW_DPX` 可取 `0` 或 `1`。默认 `FW_TILE=64`、`FW_DPX=0`。比较分块时只调整 `FW_TILE`；比较整数 DPX 时保持分块大小相同，只改变 `FW_DPX`。

DPX 用于更新对角块所在行、列以外的方块。题面保证所有有限边权是正整数。每个线程块核对两个输入块的有限距离，均不超过 2²³ 时使用 `__viaddmin_s32`。此时两段距离相加不超过 2²⁴，FP32 可以精确表示。不可达距离使用整数标记 2²⁹，两个标记相加仍在 32 位有符号整数范围内。含有更大有限距离的块使用 FP32，原输出始终以 FP32 参与最后一次取最小值。范围检查与转换耗时包含在该配置的总时间内。

## B 题与 C 题

`input`、`output`、`shape` 都是设备指针。每次调用启动一个 kernel，从 GPU 上的 `shape` 读取维度。每个输出写入一个 `int` 下标。输入值相同时选择原始下标最小的一项。连续行在 warp 内合并数值与下标，跨步归约时由相邻线程处理相邻输出列。

| 想比较的访问方式 | 调整文件顶部的值 | 受影响的公开输入 |
| --- | --- | --- |
| 每线程处理相邻两列，使用 `float2` 读取 | `ARG_COLUMNS_128` 从 `1` 改为 `2` | `(64,128,128,128), dim=2`，512 MiB |
| 每线程处理相邻两列，使用 `float2` 读取 | `ARG_COLUMNS_256` 从 `1` 改为 `2` | `(4,256,256,256), dim=1`，256 MiB |
| 一个 warp 处理一行 | `ARG_LONG_WARPS` 从 `4` 改为 `1` | `(8,1024,1024), dim=2`，32 MiB |

长度 64 的连续行始终由一个 warp 处理，每个线程读取两个值。以上调整均使用输入数据计算完整输出。B、C 的文件各自可以提交，默认源码只有 `FindMax` 的取值不同。

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

[原始五次计时及源码 SHA-256](results-1116-gpu-20261116-022947.json) 按原始字节保留，里面的 `.cu` 文件名仍指向上表的旧路径。该记录的源码哈希对应源仓库 `1d0f196`，当前迁移版本 `adc8a2b` 仅调整头文件和正无穷构造方式，不能将旧记录表述为当前源码已完成新的 GPU 回归。

本次迁移只调整目录、文件名、验证脚本的路径/编译语言参数/哈希路径和使用说明。已做静态检查，没有执行或复现 GPU 正确性实验、性能测量，也没有改动三个优化入口的计算逻辑。
