# C · Box Blur

输入是一张 FP32 灰度图。对于每个输出像素，以该像素为中心，取边长为 `kernel_size` 的正方形窗口，求窗口内有效像素的平均值。`kernel_size` 为奇数，输出图像与输入形状相同。窗口到达图像边界时，只累加图像范围内的像素，除数等于这些像素的数量。

```cpp
extern "C" void solution(const float* input_image, int kernel_size, float* output_image,
                         size_t height, size_t width);
```

| 题面图像尺寸 | 窗口边长 |
| --- | --- |
| 1920 × 1080 | 15、21、27 |
| 2048 × 2048 | 15、21、27 |

参考实现用全 1 卷积核求窗口和，再除以每个位置的有效像素数。H20 基准为 0.193130 ms，基础分 1000，最高分 1500。[接口参数与计分参数](problem.json) · [CUDA 解答](code.cpp)
