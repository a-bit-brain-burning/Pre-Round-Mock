import triton
import triton.language as tl


@triton.jit
def _row_tile(inp, out, rows, K: tl.constexpr, M: tl.constexpr, R: tl.constexpr, FIND_MAX: tl.constexpr):
    row = tl.program_id(0) * M + tl.arange(0, M)
    r = tl.arange(0, R)
    fill: tl.constexpr = -float("inf") if FIND_MAX else float("inf")
    values = tl.load(inp + row[:, None].to(tl.int64) * K + r[None, :],
                     (row[:, None] < rows) & (r[None, :] < K), fill)
    if FIND_MAX:
        best = tl.max(values, axis=1)
    else:
        best = tl.min(values, axis=1)
    # 负数优先选择第一个 NaN，非负数记录普通极值的首次下标。
    rank = tl.where(values != values, r[None, :] - K,
                    tl.where(values == best[:, None], r[None, :], K))
    first = tl.min(tl.where(r[None, :] < K, rank, K), axis=1)
    tl.store(out + row, tl.where(first < 0, first + K, first), row < rows)


@triton.jit
def _column_tile(inp, out, K: tl.constexpr, INNER: tl.constexpr, C: tl.constexpr, R: tl.constexpr,
                 FIND_MAX: tl.constexpr):
    tiles: tl.constexpr = triton.cdiv(INNER, C)
    pid = tl.program_id(0)
    outer = (pid // tiles).to(tl.int64)
    col = pid % tiles * C + tl.arange(0, C)
    r = tl.arange(0, R)
    fill: tl.constexpr = -float("inf") if FIND_MAX else float("inf")
    best = tl.full((C,), fill, tl.float32)
    index = tl.full((C,), K, tl.int32)
    for start in range(0, K, R):
        k = start + r
        values = tl.load(inp + (outer * K + k[:, None]) * INNER + col[None, :],
                         (k[:, None] < K) & (col[None, :] < INNER), fill)
        if FIND_MAX:
            local = tl.max(values, axis=0)
        else:
            local = tl.min(values, axis=0)
        rank = tl.where(values != values, k[:, None] - K,
                        tl.where(values == local[None, :], k[:, None], K))
        candidate = tl.min(tl.where(k[:, None] < K, rank, K), axis=0)
        if K <= R:
            index = candidate
        else:
            better = local > best if FIND_MAX else local < best
            take = (index == K) | ((index >= 0) & ((candidate < 0) | better))
            index = tl.where(take, candidate, index)
            best = tl.where(take, local, best)
    tl.store(out + outer * INNER + col, tl.where(index < 0, index + K, index), col < INNER)


def solution(input, dim, output, shape, ndim):
    length = input.shape[dim]
    inner = input.stride(dim)
    outputs = output.numel()
    if outputs == 0:
        return
    if inner == 1 and length <= 2048:
        rows = 32 if length <= 128 else 4
        _row_tile[(triton.cdiv(outputs, rows),)](input, output, outputs, length, rows,
                                                 triton.next_power_of_2(length), True, num_warps=4)
        return
    cols = min(128, triton.next_power_of_2(inner))
    reduction = min(256, triton.next_power_of_2(length))
    grid = outputs // inner * triton.cdiv(inner, cols)
    _column_tile[(grid,)](input, output, length, inner, cols, reduction, True, num_warps=8)
