# Day1 · 2026-11-16 模拟题

当前 `main` 分支保存原始题面和起始代码。

| 题号 | 内容 | 原始代码状态 |
| --- | --- | --- |
| [1 / A](1/problem.md) | 有向正权图的所有点对最短路 | [code.cpp](1/code.cpp)：逐中转点启动 kernel 的 Floyd 实现 |
| [2 / B](2/problem.md) | 沿指定维度返回最大值下标，并列选最小下标 | [code.cpp](2/code.cpp)：空入口模板 |
| [3 / C](3/problem.md) | 沿指定维度返回最小值下标，并列选最小下标 | [code.cpp](3/code.cpp)：空入口模板 |

优化实现位于 [csxsl 分支](https://github.com/a-bit-brain-burning/Pre-Round-Mock/tree/csxsl/Day1)，三个题目的目录和入口文件名保持一致。

## 来源

- 来源仓库：[a-bit-brain-burning/Preround](https://github.com/a-bit-brain-burning/Preround)。
- 题面与原始代码：`main@8b2795f7e846d3433b942fb88579a92d77462c56` 的 `mock/11-16/{1,2,3}/`。仅清理了第 1 题 `code.cpp` 的三处行末空白，未修改题面、接口或计算逻辑。
- csxsl 优化实现：`sim1116-csxsl@adc8a2bea4a01c9f0213569aa613b47d1d57b5e4` 的 `sim-1116/`。

本次整理只进行文件、路径、接口和 Git diff 静态检查，没有执行 GPU 正确性实验或性能测量。已有验证记录随优化实现保留在 `csxsl` 分支，来源版本和硬件边界由该分支文档说明。
