# 队友如何用 H20 模拟器估分

手里只有候选源码时，使用历史估分入口。它从同题、同语言、相同计时协议的早期 H20 回执中找参照，按用例估计耗时。手里已有正式题的逐用例测量和标程耗时，则可以直接计算预计总分和每个用例贡献的分数。

## 跑通公开示例

下面的命令从仓库根目录执行。使用 Python 3.12 和 NumPy 即可在 CPU 上估分，无需 CUDA 工具链、H20 或平台账号。

```bash
git clone --branch H20-sim --single-branch https://github.com/a-bit-brain-burning/Pre-Round-Mock.git
cd Pre-Round-Mock
python -m venv .venv
```

Linux、macOS 或 WSL 中运行 `source .venv/bin/activate`。Windows PowerShell 中运行 `.\.venv\Scripts\Activate.ps1`。接着执行：

```bash
python -m pip install -r requirements-scoring.txt
python -B h20_sim/history_model.py --history h20_sim/public_day1_history.json.gz --source h20_sim/public_argmax.cu --problem argmax --language cuda --protocol gpu-only-median-1 --as-of 2026-11-16T17:13:38.579+08:00
```

预期得到 `score = 685.0534`、`incumbent = 840.5362`、`decision = KEEP`，六个 `estimate_us` 值的单位都是微秒。公开示例使用了七个较早的 Argmax 版本，候选本身未参与校准。这一版此后实际得到 **863.0907 分**，刷新纪录 **22.5545 分**。公开样本缺少可用于核验预测区间的历史误差，因此上下限都是 `null`。这是一个点估分低于纪录、仍值得保留的真实例子。[早期公开回执](https://github.com/a-bit-brain-burning/Pre-Round-Mock/blob/97edef1bb05a3537567af6240c630147d29fa6f0/Day1/results-1116-h20.json) · [候选源码](h20_sim/public_argmax.cu) · [含候选得分的公开回执](https://github.com/a-bit-brain-burning/Pre-Round-Mock/blob/cc872b952cc2677cf7965d87e48a48f13c308aff/Day1/results-1116-h20.json)

## 换成自己的候选

1. 运行本题的正确性检查。历史估分读取源码特征，不会编译或执行候选。
2. 将 `--source` 改为候选文件路径。若手里是带 `code` 和 `sha256` 的源码快照，可改用 `--source-receipt 文件路径`。程序会核对源码摘要。
3. 将 `--history` 改为与题目匹配的校准包。随仓库提供的示例只覆盖 Day1 Argmax 和 Argmin 的 CUDA 版本。队内五场校准包文件名为 `paced-model-20261121.json.gz`，从队内取得后按本机路径传入。它包含模拟场归档，不会自动同步线上新提交。
4. 从本题回执填写 `--problem`、`--language` 和 `--protocol`。例如 Day4 的 Diagonal Matmul 使用 `diagonal-matmul`、`cuda`、`gpu-only-median-fresh-2`。将 `--as-of` 设为准备提交的时间，携带时区。回放历史版本时使用其原提交时间，只有此前已经返回的回执可参与估计。

| 输出 | 如何用于提交判断 |
| --- | --- |
| `score`、`incumbent` | 前者是点估分，后者是校准包在指定时间已知的同题最高分。`score: null` 表示缺少可用参照。 |
| `estimate_us`、`references` | 按该题用例顺序查看预计时间，再用回执编号核查参照版本。 |
| `score_low`、`score_high` | 依据历史预测误差计算的范围。上限也低于纪录时，模型可给出 `SKIP_MODEL`。两项为 `null` 时，证据不足以给出范围。 |
| `small_signal`、`gain_signal` | 分别表示值得复核的 10 分以内提升和一般加分机会。候选差距只有几分时，优先比较同形状、同输入状态的配对探针。 |
| `used_before`、`budget_decision` | 校准包有三题同场提交记录时才输出。`KEEP` 表示保留；`SKIP_RESERVE` 表示给后续候选预留名额；`SKIP_BUDGET` 表示可分配名额不足；`SKIP_REPEAT` 表示已见同源码。 |

三题共用 100 次时，区分估分和实际名额。每次 Submit 都计入队内记录，失败也占一次。命令只读取给定的历史文件，重复执行不会扣名额。进行中的新场次需要以本场记录维护已用次数。名额分配实现提供 `SubmissionPacer`，每场复用同一个实例，按候选到达顺序调用 `decide`。它从首个候选起按八小时和已见候选数开放名额。[名额分配](h20_sim/submission_pacing.py)

## 正式题的逐用例计分

复制 [逐用例输入示例](h20_sim/public_formal_card.json) 为自己的 JSON。原文件使用已结束的 Day1 公开回执演示字段和单位。直接运行以下命令，会列出六个用例并返回 `AWAITING_BENCHMARKS`。

```bash
python -B h20_sim/contest_scoring.py h20_sim/public_formal_card.json
```

| 字段 | 正式提交前填写的内容 |
| --- | --- |
| `candidate_sha256` | 候选源码原始字节的 SHA-256。Linux 可用 `sha256sum`，PowerShell 可用 `Get-FileHash -Algorithm SHA256`。 |
| `protocol`、`cases[].context` | 使用本场实际计时协议。每例填写算子、公开形状、数据类型、布局、输入状态和估计方法。每例的协议须与文件顶层一致。 |
| `rule` | 正式计分使用 `preliminary_per_case`。填写官方题目基础分 `base_points`，在 `case_ids` 中按顺序列全用例，顺序须与 `cases` 完全一致。 |
| `measurement_us`、`benchmark_us` | 使用 `paired_measurement` 时，前者填提交前取得的对应测量，后者填官方公布的同一用例标程耗时。两者都使用微秒。毫秒乘 1,000。缺少任一标程耗时时只报告用时。 |
| `incumbent`、`calibration` | 填写当时同题最高分及已有校准记录。首次估计可用空校准列表。对照记录必须来自更早返回的其他源码，并匹配用例及计时条件。 |

对正耗时用例，程序计算 `min(1.5, benchmark_us / estimate_us)`，对全部用例取平均后乘以基础分。结果中的 `cases[].score_contribution` 是每例对总分的贡献；`score_change` 是预计总分减去已有最高分。历史源码入口使用模拟场的总耗时公式。正式题需要相应的逐用例标程耗时和候选测量。[计分实现](h20_sim/contest_scoring.py) · [H20 探针参数](h20_sim/contest_probe.json)

## 需要重放 kernel 时

[SM90 编译与 CPU 重放](h20_sim/run_suite.py)、[性能分析](h20_sim/run_performance.py) 和 [Accel-Sim 轨迹工具](common/simulator006.py) 随分支提供。完整重放需要 Linux 或 WSL 的 PyTorch、Triton、编译工具，以及单独的生产算子工作副本；将 `PREROUND_OPERATOR_ROOT` 指向其中包含 `common/validation.py` 和 `SwiGLU_backward/solution.py` 的目录。编译后端版本见 [backend.lock.json](h20_sim/backend.lock.json)。上面的历史估分和独立计分命令可以直接从本分支运行。
