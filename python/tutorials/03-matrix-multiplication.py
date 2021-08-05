"""
Matrix Multiplication
======================
In this tutorial, you will write a 25-lines high-performance FP16 matrix multiplication
kernel that achieves performance on par with cuBLAS.
You will specifically learn about:

- Block-level matrix multiplications
- Multi-dimensional pointer arithmetic
- Program re-ordering for improved L2 cache hit rate
- Automatic performance tuning
"""

# %%
# Motivations
# -------------
# Matrix multiplications are a key building block of most modern high-performance computing systems.
# They are notoriously hard to optimize, hence their implementation is generally done by
# hardware vendors themselves as part of so-called "kernel libraries" (e.g., cuBLAS).
# Unfortunately, these libraries are often proprietary and cannot be easily customized
# to accomodate the needs of modern deep learning workloads (e.g., fused activation functions).
# In this tutorial, you will learn how to implement efficient matrix multiplications by
# yourself with Triton, in a way that is easy to customize and extend.
#
# Roughly speaking, the kernel that we will write will implement the following blocked
# algorithm to multiply a (MxK) by a (KxN) matrix:
#
#  .. code-block:: python
#
#    # do in parallel
#    for m in range(0, M, BLOCK_SIZE_M):
#      # do in parallel
#      for n in range(0, N, BLOCK_SIZE_N):
#        acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
#        for k in range(0, K, BLOCK_SIZE_K):
#          a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
#          b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
#          acc += dot(a, b)
#        C[m : m+BLOCK_SIZE_M, n : n+BLOCK_SIZE_N] = acc;
#
# where each iteration of the doubly-nested for-loop corresponds to a Triton program instance.

# %%
# Compute Kernel
# ----------------
#
# The above algorithm is, actually, fairly straightforward to implement in Triton.
# The main difficulty comes from the computation of the memory locations at which blocks
#  of :code:`A` and :code:`B` must be read in the inner loop. For that, we need
# multi-dimensional pointer arithmetics.
#
# Pointer Arithmetics
# ~~~~~~~~~~~~~~~~~~~~
#
# For a row-major 2D tensor :code:`X`, the memory location of :code:`X[i, j]` is given b
# y :code:`&X[i, j] = X + i*stride_x_0 + j*stride_x_1`.
# Therefore, blocks of pointers for :code:`A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K]` and
# :code:`B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]` can be defined in pseudo-code as:
#
#  .. code-block:: python
#
#    &A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K] =  A + (m : m+BLOCK_SIZE_M)[:, None]*A.stride(0) + (k : k+BLOCK_SIZE_K)[None, :]*A.stride(1);
#    &B[k : k+BLOCK_SIZE_K, n:n+BLOCK_SIZE_N] =  B + (k : k+BLOCK_SIZE_K)[:, None]*B.stride(0) + (n : n+BLOCK_SIZE_N)[None, :]*B.stride(1);
#
# Which means that pointers for blocks of A and B can be initialized (i.e., :code:`k=0`) in Triton as:
#
#  .. code-block:: python
#
#    pid_m = triton.program_id(0)
#    pid_n = triton.program_id(1)
#    rm = pid_m * BLOCK_SIZE_M + triton.arange(0, BLOCK_SIZE_M)
#    rn = pid_n * BLOCK_SIZE_N + triton.arange(0, BLOCK_SIZE_N)
#    rk = triton.arange(0, BLOCK_SIZE_K)
#    // pointer for A operand
#    pa = A + (rm[:, None] * stride_a_0 + rk[None, :] * stride_a_1);
#    // pointer for B operand
#    pb = B + (rk[:, None] * stride_b_0 + rn[None, :] * stride_b_1);
#
# And then updated in the inner loop as follows:
#
#  .. code-block:: python
#
#    pa += BLOCK_SIZE_K * stride_a_1;
#    pb += BLOCK_SIZE_K * stride_b_0;
#
#
# L2 Cache Optimizations
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# As mentioned above, each program instance computes a :code:`[BLOCK_SIZE_M, BLOCK_SIZE_N]`
#  block of :code:`C`.
# It is important to remember that the order in which these blocks are computed does
# matter, since it affects the L2 cache hit rate of our program. and unfortunately, a
# a simple row-major ordering
#
#  .. code-block:: Python
#
#    pid = triton.program_id(0);
#    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M;
#    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N;
#    pid_m = pid / grid_n;
#    pid_n = pid % grid_n;
#
# is just not going to cut it.
#
# One possible solution is to launch blocks in an order that promotes data reuse.
# This can be done by 'super-grouping' blocks in groups of :code:`GROUP_M` rows before
# switching to the next column:
#
#  .. code-block:: python
#
#    pid = triton.program_id(0);
#    width = GROUP_M * grid_n;
#    group_id = pid // width;
#    # we need to handle the case where M % (GROUP_M*BLOCK_SIZE_M) != 0
#    group_size = min(grid_m - group_id * GROUP_M, GROUP_M);
#    pid_m = group_id * GROUP_M + (pid % group_size);
#    pid_n = (pid % width) // (group_size);

# For example, in the following matmul where each matrix is 9 blocks by 9 blocks,
# we can see that if we compute the output in row-major ordering, we need to load 90
# blocks into SRAM to compute the first 9 output blocks, but if we do it in grouped
# ordering, we only need to load 54 blocks.
#   .. image:: grouped_vs_row_major_ordering.png
#
# In practice, this can improve the performance of our matrix multiplication kernel by
# more than 10\% on some hardware architecture (e.g., 220 to 245 TFLOPS on A100).
#

# %%
# Final Result
# -------------
#

import torch
import triton
import triton.language as tl

# %
# :code:`triton.jit`'ed functions can be auto-tuned by using the `triton.autotune`
# decorator, which consumes:
#   - A list of :code:`triton.Config` objects that define different configurations of
#       meta-parameters (e.g., BLOCK_SIZE_M) and compilation options (e.g., num_warps) to try
#   - An autotuning *key* whose change in values will trigger evaluation of all the
#       provided configs

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64 , 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64 , 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64 , 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32 , 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64 , 'BLOCK_SIZE_N': 32 , 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32 , 'BLOCK_SIZE_N': 64 , 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
# %
# We can now define our kernel as normal, using all the techniques presented above
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    **meta,
):
    """Kernel for computing the matmul AB = C

    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # extract meta-parameters
    BLOCK_SIZE_M = meta['BLOCK_SIZE_M']
    BLOCK_SIZE_N = meta['BLOCK_SIZE_N']
    BLOCK_SIZE_K = meta['BLOCK_SIZE_K']
    GROUP_SIZE_M = 8
    pid = tl.program_id(axis=0)

    # the number of blocks is the ceil(M / BLOCK_SIZE_M) since we need an extra block
    # Note that this will lead to some quantization in performance where time-taken jumps
    # when you need to add a new block
    n_blocks_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    n_blocks_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N

    # Map PIDs to the block they should compute. This is done in a grouped ordering
    # to promote L2 cache reuse.
    n_output_blocks_in_group = GROUP_SIZE_M * n_blocks_n
    group_id = pid // n_output_blocks_in_group
    first_m_block_in_group = group_id * GROUP_SIZE_M

    # If the number of blocks is not divisible by the group size, the last group is smaller
    group_size_m = min(n_blocks_m - first_m_block_in_group, GROUP_SIZE_M)

    # Within a group, we compute in col-major ordering, block_m and block_n are the
    # output row and col that this program is computing in terms of blocks
    block_m = first_m_block_in_group + (pid % group_size_m)
    block_n = (pid % n_output_blocks_in_group) // group_size_m

    # Convert from block indices back to element indices
    m_start = block_m * BLOCK_SIZE_M
    n_start = block_n * BLOCK_SIZE_N

    # Expand out to all the offsets for each of the elements in this block.
    m_offsets_a = (m_start + tl.arange(0, BLOCK_SIZE_M))[:, None]
    n_offsets_b = (n_start + tl.arange(0, BLOCK_SIZE_N))[None, :]
    k_offsets = tl.arange(0, BLOCK_SIZE_K)

    # Get the pointers for the first block of each. We will advance this pointer
    # as we move in the K direction and accumulate.
    # a_ptrs should contain BLOCK_SIZE_M * BLOCK_SIZE_K pointers
    a_ptrs = a_ptr + (stride_am * m_offsets_a + stride_ak * k_offsets[None, :])
    # b_ptrs should contain BLOCK_SIZE_K * BLOCK_SIZE_N pointers
    b_ptrs = b_ptr + (stride_bk * k_offsets[:, None] + stride_bn * n_offsets_b)
    # We accumulate internally in fp32, but the output is written out in the dtype
    # of the tensor when it is stored
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        # Note that for simplicity, we don't apply a mask here. This means that if K is
        # not a multiple of BLOCK_SIZE_K, this will access out-of-bounds memory and
        # accumulate it incorrectly.
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # We accumulate along the K dimension
        accumulator += tl.dot(a, b)

        # Advance the ptrs to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # triton can accept arbitrary activation function via metaparameters!
    if meta['ACTIVATION']:
        accumulator = meta['ACTIVATION'](accumulator)

    m_offsets_c = (m_start + tl.arange(0, BLOCK_SIZE_M))[:, None]
    n_offsets_c = (n_start + tl.arange(0, BLOCK_SIZE_N))[None, :]
    c_ptrs = c_ptr + stride_cm * m_offsets_c + stride_cn * n_offsets_c
    mask = (m_offsets_c < M) & (n_offsets_c < N)
    tl.store(c_ptrs, accumulator, mask=mask)


# we can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `_matmul`
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


# %%
# We can now create a convenience wrapper function that only takes two input tensors
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel


def matmul(a, b, activation=None):
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    assert a.is_contiguous(), "matrix A must be contiguous"
    assert b.is_contiguous(), "matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    assert (
        K % 32 == 0
    ), "We don't check memory-out-of-bounds with K so K must be divisible by BLOCK_SIZE_K"
    # allocates output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        ACTIVATION=activation,
    )
    return c


# %%
# Unit Test
# -----------
#
# We can test our custom matrix multiplication operation against a native torch implementation (i.e., cuBLAS)

torch.manual_seed(0)
a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
triton_output = matmul(a, b, activation=None)
torch_output = torch.matmul(a, b)
print(f"{triton_output=}")
print(f"{torch_output=}")
if triton.testing.allclose(triton_output, torch_output):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

# %%
# Benchmark
# --------------
#
# Square Matrix Performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# We can now compare the performance of our kernel against that of cuBLAS. Here we focus on square matrices, but feel free to arrange this script as you wish to benchmark any other matrix shape.


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # argument names to use as an x-axis for the plot
        x_vals=[
            128 * i for i in range(1, 33)
        ],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        # possible values for `line_arg``
        line_vals=['cublas', 'cublas + relu', 'triton', 'triton + relu'],
        # label name for the lines
        line_names=["cuBLAS", "cuBLAS (+ torch.nn.LeakyReLU)", "Triton", "Triton (+ LeakyReLU)"],
        # line styles
        styles=[('green', '-'), ('green', '--'), ('blue', '-'), ('blue', '--')],
        ylabel="TFLOPS",  # label name for the y-axis
        plot_name="matmul-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={},
    )
)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b))
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b))
    if provider == 'cublas + relu':
        torch_relu = torch.nn.ReLU(inplace=True)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch_relu(torch.matmul(a, b))
        )
    if provider == 'triton + relu':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul(a, b, activation=leaky_relu)
        )
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True)
