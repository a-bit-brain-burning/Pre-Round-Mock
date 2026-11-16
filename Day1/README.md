# Day1 · CUDA 模拟题最优实测版本

当前 `Fomak` 分支保存三个可独立提交的源码，以及验证程序和精简成绩记录。

| 题目 | 源码 | 状态 |
| --- | --- | --- |
| A · 全源最短路 | [1/code.cpp](1/code.cpp) | 已完成；本轮没有改动，19 项本机正确性检查通过 |
| B · argmax | [2/code.cpp](2/code.cpp) | H20 正式评测 6/6，通过时间 308.704 μs，**955.2419 分** |
| C · argmin | [3/code.cpp](3/code.cpp) | H20 正式评测 6/6，通过时间 308.864 μs，**954.7471 分** |

两题各通过 357 项本机正确性检查，SM90/SM120 编译通过。保留 CPU/GPU shape 支持、并列取首个索引、NaN、正负零、无穷值、非对齐输入和输出边界处理。每个源码均提供题目要求的 `extern "C" void solution(...)` 入口。

第 2、3 组使用独立 L2 持久缓存窗口，并在其他尺寸恢复初始配额。第 2 组采用 float2 CG 流式读取，第 5 组使用四 CTA cluster 默认调度。第 4 组对两个独立列块分别同步；argmin 的非输出 warp 到达屏障后退出。

**960 分尚未达到，已按用户要求停止优化。** 所选源码各有一次正式成绩，细小差异不代表稳定加速。缓存优化的收益依赖评测时的缓存状态，不能视作完全冷缓存带宽的同等提升。

[best-results.json](best-results.json) 包含完整任务号、源码 SHA-256、六组耗时和缓存配置；[verification-results.json](verification-results.json) 保存对应正确性记录。完整历史实验和日志保留在本地分支 `Fomak-local-experiments-c214f9d`，未包含在此次远程提交中。

本机验证使用 CUDA 12.8.93。安装支持相应架构的 CUDA 工具链后，可运行：

```bash
nvcc -O2 -std=c++20 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_120,code=sm_120 -x cu verify.cu -o /tmp/verify-day1
/tmp/verify-day1 --reductions
/tmp/verify-day1 --specialized
```

本机 RTX 5070 Laptop 的耗时仅用于正确性诊断；表中性能来自真实 H20 正式评测。
