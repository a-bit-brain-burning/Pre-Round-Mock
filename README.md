# Pre-Round Mock

初赛 CUDA 算子模拟题练习仓库，按练习日整理题面、原始代码与队员实现。

| 分支 | 内容 |
| --- | --- |
| [main](https://github.com/a-bit-brain-burning/Pre-Round-Mock/tree/main) | 原始题面与起始代码 |
| [csxsl](https://github.com/a-bit-brain-burning/Pre-Round-Mock/tree/csxsl) | csxsl 的优化实现、验证脚本及已有测量记录 |

两条分支使用相同的题目路径。`main` 中的 `code.cpp` 是原始版本，`csxsl` 中同路径的 `code.cpp` 是优化版本；使用 `git diff main..csxsl -- Day1` 可查看差异。

## Day1 · 2026-11-16

B、C 当前记录的最高 H20 正式成绩分别为 863.0907 分和 859.5084 分，均通过全部六组测试。比赛平台选择 CUDA C++，每题提交一份文件。[B 提交源码](https://github.com/a-bit-brain-burning/Pre-Round-Mock/blob/fc92a86c32a70901cc3927f869a72111c33e9b9a/Day1/2/code.cpp) · [C 提交源码](https://github.com/a-bit-brain-burning/Pre-Round-Mock/blob/fc92a86c32a70901cc3927f869a72111c33e9b9a/Day1/3/code.cpp) · [逐轮方法和效果](Day1/H20-reduction-experiments.md) · [分组耗时与提交编号](Day1/results-1116-h20.json)

| 题号 | 题目 | 题面 | 代码 |
| --- | --- | --- | --- |
| 1 / A | All Pairs Shortest Path | [problem.md](Day1/1/problem.md) | [code.cpp](Day1/1/code.cpp) |
| 2 / B | Argmax Over Dimension | [problem.md](Day1/2/problem.md) | [code.cpp](Day1/2/code.cpp) |
| 3 / C | Argmin Over Dimension | [problem.md](Day1/3/problem.md) | [code.cpp](Day1/3/code.cpp) |

题目说明与来源见 [Day1/README.md](Day1/README.md)。源码中的 `solution(...)` 为比赛平台入口；`code.cpp` 内包含 CUDA 代码，需要使用 CUDA C++ 工具链。

本仓库从 [Preround](https://github.com/a-bit-brain-burning/Preround) 中独立整理，不迁入其余正式题、模拟器、缓存和旧 Git 历史。原贡献的来源版本记录在 Day1 文档中；迁移提交不代表重新开发这些实现。
