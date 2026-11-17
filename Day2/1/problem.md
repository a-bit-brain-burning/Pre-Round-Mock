# A · 3D Average Pooling

输入是按行存储的 FP32 张量，形状为 `(H, W, D)`，最后一个维度连续。给定立方窗口边长 `kernel_size`、步长 `stride` 和六个面各自的填充宽度 `padding`，计算每个窗口的平均值。

令 $k$、$s$、$p$ 分别表示这三个参数。输入某个维度的长度为 $L$ 时，对应的输出长度为 $\lfloor (L+2p-k)/s \rfloor + 1$。输出位置 $(i,j,l)$ 对应的输入窗口起点为 $(is-p,js-p,ls-p)$。窗口超出输入边界的位置取零，所有窗口均除以 $k^3$。

```cpp
extern "C" void solution(const float* input, int kernel_size, int stride, int padding,
                         float* output, size_t H, size_t W, size_t D);
```

| 输入形状 | 窗口边长 | 步长 | 填充宽度 |
| --- | ---: | ---: | ---: |
| 192 × 192 × 192 | 5 | 2 | 2 |
| 224 × 224 × 224 | 7 | 3 | 3 |
| 512 × 512 × 512 | 3 | 3 | 1 |
| 784 × 784 × 784 | 5 | 2 | 2 |
| 1024 × 1024 × 1024 | 3 | 3 | 1 |

平台参考使用 `torch.nn.functional.avg_pool3d`。H20 基准为 1.915324 ms，基础分 1000，最高分 1500。[接口参数与计分参数](problem.json) · [CUDA 解答](code.cpp)
