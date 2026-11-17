# B · Batch Normalization

输入和输出均为形状 `(B, F, D1, D2)` 的 FP32 张量。每个元素减去均值，再除以 $\sqrt{\mathrm{Var}+10^{-5}}$。方差采用总体方差，归一化后直接输出结果。

英文题面写明每个空间位置沿 `B` 单独计算统计量。随题提供的参考实现调用 `nn.BatchNorm2d(F, affine=False, track_running_stats=False, eps=1e-5)`，对每个通道在 `(B, D1, D2)` 上共同计算均值和方差。平台 Run 样例的首个输入为 1，该位置沿 `B` 的两个值为 1 和 −1，仅沿 `B` 计算约为 0.999995，平台预期为 0.886794。当前解答采用参考实现的通道统计维度。[接口参数与计分参数](problem.json)

```cpp
extern "C" void solution(const float* X, float* Y, size_t B, size_t F, size_t D1, size_t D2);
```

| B | F | D1 | D2 |
| ---: | ---: | ---: | ---: |
| 16 | 64 | 256 | 256 |
| 32 | 128 | 128 | 128 |
| 8 | 256 | 64 | 64 |
| 4 | 32 | 512 | 512 |

H20 基准为 0.387001 ms，基础分 1000，最高分 1500。[CUDA 解答](code.cpp)
