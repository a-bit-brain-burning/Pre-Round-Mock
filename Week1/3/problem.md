# Argmin Over Dimension

practice-2026-11-16 · Problem C

| Contest base points | Maximum points | Official baseline total (T_std) |
| :--- | :--- | :--- |
| **1000** | **1500** | **0.294887 ms** |

Points = base × min(1.5, T_std / total GPU time), rounded to 4 decimal places. Every submission is scored under the current T_std, so a T_std change rescores them all.

[Easy] [My submissions] [Leaderboard] [Rules]

Find the indices of minimum values along a specified dimension of an input tensor:

\[
\text{output}[i_1, \dots, i_{d-1}, i_{d+1}, \dots, i_n] = \arg\min_{i_d} \text{input}[i_1, \dots, i_d, \dots, i_n]
\]

where \(d\) is the dimension to perform argmin over, \(n\) is the number of dimensions.

### Input:
*   Tensor `input` of arbitrary shape \(S_1 \times S_2 \times \cdots \times S_n\)
*   `dim` (\(d\)): Dimension to perform argmin over (0-based indexing)
*   `shape` : Array containing the dimensions of the input tensor
*   `ndim` (\(n\)): Number of dimensions in the input tensor

### Output:
*   Tensor `output` with shape \(S_1 \times \cdots \times S_{d-1} \times S_{d+1} \times \cdots \times S_n\)
    *   The output contains indices of minimum values along the specified dimension
    *   The dimension being reduced is removed from the output shape

### Notes:
*   The input tensor is stored in row-major order
*   In case of ties (multiple minimum values), return the index of the first occurrence
*   The output indices are 0-based
*   The output tensor has one fewer dimension than the input tensor
*   This problem is adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench)

### Test Case Sizes
*   `shape=(16, 128, 256), dim=1`
*   `shape=(32, 512, 512), dim=0`
*   `shape=(8, 1024, 1024), dim=2`
*   `shape=(64, 128, 128, 128), dim=2`
*   `shape=(4, 256, 256, 256), dim=1`
*   `shape=(128, 64, 64, 64), dim=3`
