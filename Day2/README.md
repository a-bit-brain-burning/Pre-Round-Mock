# Day2 · 2026-11-17

[Day2 实验整理](day2-review.md) 记录三题的 H20 成绩、各组尺寸的优化效果，以及正式初赛五题的验证方法。

| 题目 | 本目录入口 | 主要实验 |
| --- | --- | --- |
| [A · 3D Average Pooling](1/problem.md) | [v45 CUDA](1/code.cpp) | [显存流量](1/pool-traffic.cu)、[小尺寸分块](1/pool-small-tuned.cu)、[连续视图](1/pool-views.cu)、[Triton](1/pool-triton.py) |
| [B · Batch Normalization](2/problem.md) | [v2 CUDA](2/code.cpp) | [分组归约](2/bn-group.cu)、[cluster](2/bn-cluster.cu)、[TMA](2/bn-bulk.cu)、[逐通道检查输入变化](2/bn-channel-cache.cu) |
| [C · Box Blur](3/problem.md) | [v4 满分 CUDA](3/code.cpp) | 寄存器滚动求和、每线程处理相邻四列、图像边界计数 |

[H20 实验索引](experiment-index.json) 收录本工作区的 Run、Submit 和已归档的队内纪录。每次尝试可按题号、版本备注和 SHA-256 查找。Submit 含平台返回的分组耗时，诊断 Run 保留独立测量的数据行，单位见版本备注与源码。索引中的源码路径以本目录为起点。队内下载的归档在本机保存。

本机使用 PyTorch 及 CUDA 核对数值。[三个算子](verify.py) · [连续视图](verify-views.py) · [同址改写与重新分配](verify-norm-update.py)。复现命令见 [实验整理](day2-review.md)。

[评测工具](../evaluate.ps1) 要求通过 `-Note` 填写版本及优化动作，支持独立复测与继续处理已有任务。[成绩同步](../sync_team_best.ps1) 按题目核对提交编号和 SHA-256。运行前设置环境变量 `MOCK_RELAY_URL` 为队内服务的 HTTPS 地址，`MOCK_RELAY_LOGIN_FILE` 为仓库外登录 JSON 文件的绝对路径。
