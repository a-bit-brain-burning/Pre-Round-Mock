# Pre-Round Mock

本分支提供 H20 离线模拟器。队友可从 [H20 提交前估分指南](H20_SCORING.md) 开始，使用仓库自带的数据运行一次完整估分，然后替换为自己的源码。常用入口为 [源码估分](h20_sim/history_model.py) 和 [正式赛逐用例计分](h20_sim/contest_scoring.py)。

初赛 CUDA 算子模拟题练习仓库，按练习日整理题面、原始代码与队员实现。

| 分支 | 内容 |
| --- | --- |
| [main](https://github.com/a-bit-brain-burning/Pre-Round-Mock/tree/main) | 原始题面与起始代码 |
| [csxsl](https://github.com/a-bit-brain-burning/Pre-Round-Mock/tree/csxsl) | csxsl 的优化实现、验证脚本及已有测量记录 |
| H20-sim | H20 模拟器、公开校准示例、逐用例估分与三题提交名额分配 |

`main` 与 `csxsl` 使用相同的题目路径。使用 `git diff main..csxsl -- Day1` 可比较起始代码和优化版本。

## Day1 · 2026-11-16

| 题号 | 题目 | 题面 | 代码 |
| --- | --- | --- | --- |
| 1 / A | All Pairs Shortest Path | [problem.md](Day1/1/problem.md) | [code.cpp](Day1/1/code.cpp) |
| 2 / B | Argmax Over Dimension | [problem.md](Day1/2/problem.md) | [code.cpp](Day1/2/code.cpp) |
| 3 / C | Argmin Over Dimension | [problem.md](Day1/3/problem.md) | [code.cpp](Day1/3/code.cpp) |

题目说明与来源见 [Day1/README.md](Day1/README.md)。源码中的 `solution(...)` 为比赛平台入口；`code.cpp` 内包含 CUDA 代码，需要使用 CUDA C++ 工具链。

练习题从 [Preround](https://github.com/a-bit-brain-burning/Preround) 整理而来，原贡献的来源版本记录在 Day1 文档中。
