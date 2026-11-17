import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _horizontal_five(x):
    rows = tl.arange(0, 16)[:, None] + tl.full((16, 128), 0, tl.int32)
    pairs = tl.sum(tl.reshape(x, (16, 2, 128)), 1)
    width = pairs + tl.gather(pairs, (rows + 1) % 16, 0) + tl.gather(x, (2 * rows + 4) % 32, 0)
    columns = tl.arange(0, 64)[None, :] + tl.full((16, 64), 0, tl.int32)
    pairs_d = tl.sum(tl.reshape(width, (16, 64, 2)), 2)
    return (tl.gather(pairs_d, (columns + 1) % 64, 1)
            + tl.gather(pairs_d, (columns + 2) % 64, 1)
            + tl.gather(width, (2 * columns + 6) % 128, 1))


@triton.jit
def _horizontal_seven(x):
    rows = tl.arange(0, 8)[:, None] + tl.full((8, 128), 0, tl.int32)
    width = tl.full((8, 128), 0, tl.float32)
    for i in tl.static_range(7):
        width += tl.gather(x, (3 * rows + i) % 32, 0)
    columns = tl.arange(0, 64)[None, :] + tl.full((8, 64), 0, tl.int32)
    result = tl.full((8, 64), 0, tl.float32)
    for i in tl.static_range(7):
        result += tl.gather(width, (3 * columns + i + 1) % 128, 1)
    return result


@triton.jit
def _plane(desc, h, w, d):
    return tl.reshape(desc.load([h, w, d]), (32, 128))


@triton.jit
def _overlap(desc, output, O: tl.constexpr, S: tl.constexpr, TH: tl.constexpr, TW: tl.constexpr, TD: tl.constexpr):
    ht, wt, dt = tl.program_id(2), tl.program_id(1), tl.program_id(0)
    h0, w0, d0 = ht * TH * S - S, wt * TW * S - S, dt * TD * S - 4
    older = _plane(desc, h0, w0, d0)
    newer = _plane(desc, h0 + S, w0, d0)
    for j in tl.static_range(1, S):
        older += _plane(desc, h0 + j, w0, d0)
        newer += _plane(desc, h0 + S + j, w0, d0)
    for i in range(TH):
        h = h0 + (i + 2) * S
        edge = _plane(desc, h, w0, d0)
        values = (older + newer) + edge
        if S == 2:
            sums = _horizontal_five(values)
            rw = tl.arange(0, 16)
        else:
            sums = _horizontal_seven(values)
            rw = tl.arange(0, 8)
        rd = tl.arange(0, 64)
        oh, ow, od = ht * TH + i, wt * TW + rw, dt * TD + rd
        valid = (oh < O) & (rw[:, None] < TW) & (ow[:, None] < O) & (rd[None, :] < TD) & (od[None, :] < O)
        tl.store(output + (oh * O + ow[:, None]) * O + od[None, :], sums / ((2*S+1)**3), valid)
        older = newer
        newer = edge
        if i + 1 < TH:
            for j in tl.static_range(1, S):
                newer += _plane(desc, h + j, w0, d0)


@triton.jit
def _disjoint(input, output, N: tl.constexpr, O: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr):
    oh = tl.program_id(1)
    ow = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    v = tl.arange(0, N)
    h0, w0 = oh * 3 - 1, ow * 3 - 1
    total = tl.full((ROWS, N), 0, tl.float32)
    for i in tl.static_range(3):
        for j in tl.static_range(3):
            h, w = h0 + i, w0 + j
            valid = (h >= 0) & (h < N) & (w[:, None] >= 0) & (w[:, None] < N)
            total += tl.load(input + (h * N + w[:, None]) * N + v[None, :], valid, 0, cache_modifier=".cg")
    d = tl.arange(0, COLS)[None, :] + tl.full((ROWS, COLS), 0, tl.int32)
    center = 3 * d
    left = tl.gather(total, tl.maximum(center - 1, 0) % N, 1)
    middle = tl.gather(total, center % N, 1)
    right = tl.gather(total, (center + 1) % N, 1)
    result = (tl.where(center > 0, left, 0) + middle + tl.where(center + 1 < N, right, 0)) / 27
    tl.store(output + (oh * O + ow[:, None]) * O + d, result, (ow[:, None] < O) & (d < O))


@triton.jit
def _direct(input, output, H: tl.constexpr, W: tl.constexpr, D: tl.constexpr, K: tl.constexpr,
            S: tl.constexpr, P: tl.constexpr, OW: tl.constexpr, OD: tl.constexpr, COUNT: tl.constexpr):
    index = tl.program_id(0) * 128 + tl.arange(0, 128)
    h0, w0, d0 = index // (OW * OD) * S - P, index // OD % OW * S - P, index % OD * S - P
    result = tl.full((128,), 0, tl.float32)
    for i in range(K):
        for j in range(K):
            for l in range(K):
                h, w, d = h0 + i, w0 + j, d0 + l
                valid = (index < COUNT) & (h >= 0) & (h < H) & (w >= 0) & (w < W) & (d >= 0) & (d < D)
                result += tl.load(input + (h.to(tl.int64) * W + w) * D + d, valid, 0)
    tl.store(output + index, result / (K * K * K), index < COUNT)


def solution(input, kernel_size, stride, padding, output, H, W, D):
    if kernel_size <= 0 or stride <= 0 or padding < 0:
        return
    oh = (H + 2 * padding - kernel_size) // stride + 1
    ow = (W + 2 * padding - kernel_size) // stride + 1
    od = (D + 2 * padding - kernel_size) // stride + 1
    if min(oh, ow, od) <= 0:
        return
    if H == W == D and kernel_size == 3 and stride == 3 and padding == 1 and D in (512, 1024):
        _disjoint[(triton.cdiv(ow, 4), oh)](input, output, D, od, 4, triton.next_power_of_2(od), num_warps=8)
        return
    if H == W == D and (D, kernel_size, stride, padding) in ((192, 5, 2, 2), (224, 7, 3, 3), (784, 5, 2, 2)) and input.data_ptr() % 16 == 0:
        desc = TensorDescriptor(input, [H, W, D], [W*D, D, 1], [1, 32, 128])
        th, tw, td = (8, 14, 56) if stride == 2 else (3, 8, 40)
        _overlap[(triton.cdiv(od, td), triton.cdiv(ow, tw), triton.cdiv(oh, th))](
            desc, output, od, stride, th, tw, td, num_warps=8)
        return
    count = oh * ow * od
    _direct[(triton.cdiv(count, 128),)](input, output, H, W, D, kernel_size, stride, padding, ow, od, count)
