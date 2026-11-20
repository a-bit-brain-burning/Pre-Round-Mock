# Pre-Round Mock

初赛 CUDA 算子模拟题练习仓库，按练习日整理题面、原始代码与队员实现。

| 分支 | 内容 |
| --- | --- |
| [main](https://github.com/a-bit-brain-burning/Pre-Round-Mock/tree/main) | 原始题面与起始代码 |
| [csxsl](https://github.com/a-bit-brain-burning/Pre-Round-Mock/tree/csxsl) | csxsl 的优化实现、验证脚本及已有测量记录 |

两条分支使用相同的题目路径。`main` 中的 `code.cpp` 是原始版本，`csxsl` 中同路径的 `code.cpp` 是优化版本；使用 `git diff main..csxsl -- Day1` 可查看差异。

## 用 AKO4X 逐题优化

把 AKO4X 和本仓库放在相邻目录，并登录要用的 Codex CLI 或 Claude Code。设置 `PREROUND_RELAY_URL` 为队内 relay 的 HTTPS 地址，或将地址写入本地 `Day2/.cache/relay/base-url`。启动器复用本地 relay 会话，会话需要更新时会打开交互式登录。

每题开一个终端，在现有 `DayN/题号` 目录中启动。以下三条命令分别在三个终端运行。

```bash
cd ~/Pre-Round-Mock/Day1/1 && ../../ako4x --agent codex
cd ~/Pre-Round-Mock/Day1/2 && ../../ako4x --agent claude
cd ~/Pre-Round-Mock/Day1/3 && ../../ako4x --agent codex --worker-agent claude
```

主控在当前终端显示进度、接收你的纠正，并分派 worker 构建候选。你可以直接告诉主控切换方向、先检查正确性或保留当前锚点。也可在仓库根目录指定 `./ako4x --day Day1 --problem A`。AKO4X 默认从相邻的 `~/AKO4X` 读取，其他位置可设 `AKO4X_ROOT`。

同一题退出后，用同一条命令或换另一种 `--agent` 可从已有轮次、已评测版本和经验继续。默认最多两轮，每个进入正式 A/B 的候选最多使用五对 Submit。三题共享评测队列。`--submit-pairs 0` 可只做 Run。每轮五段经验与源码、评测证据一起归档，可复用的结论写入 `TriedMethods.md` 和 `kb/attempts/`。

## Day1 · 2026-11-16

| 题号 | 题目 | 题面 | 代码 |
| --- | --- | --- | --- |
| 1 / A | All Pairs Shortest Path | [problem.md](Day1/1/problem.md) | [code.cpp](Day1/1/code.cpp) |
| 2 / B | Argmax Over Dimension | [problem.md](Day1/2/problem.md) | [code.cpp](Day1/2/code.cpp) |
| 3 / C | Argmin Over Dimension | [problem.md](Day1/3/problem.md) | [code.cpp](Day1/3/code.cpp) |

题目说明与来源见 [Day1/README.md](Day1/README.md)。源码中的 `solution(...)` 为比赛平台入口；`code.cpp` 内包含 CUDA 代码，需要使用 CUDA C++ 工具链。

本仓库从 [Preround](https://github.com/a-bit-brain-burning/Preround) 中独立整理，不迁入其余正式题、模拟器、缓存和旧 Git 历史。原贡献的来源版本记录在 Day1 文档中；迁移提交不代表重新开发这些实现。
