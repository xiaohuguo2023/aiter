from typing import Optional
import torch
import triton
import triton.language as tl
from aiter.ops.triton.utils.pid_preprocessing import pid_grid, remap_xcd


@triton.heuristics({
    'EVEN_K':lambda args: args['K'] % args['BLOCK_SIZE_K'] == 0, 
    'GRID_MN':lambda args: triton.cdiv(args['M'], args['BLOCK_SIZE_M']) * triton.cdiv(args['N'], args['BLOCK_SIZE_N'])
})
@triton.jit
def _gemm_a16_w16_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    remap_xcd(pid, GRID_MN)

    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)

    tl.assume(pid_m > 0)
    tl.assume(pid_n > 0)

    # Create pointers for first block of A and B input matrices
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        accumulator += tl.dot(a, b, input_precision="ieee")

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(c_ptr.type.element_ty)

    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# Wrapper for gemm kernel.
def gemm_a16w16(x, 
                w, 
                dtype: Optional[float] = torch.bfloat16,
                ):
    """
    Computes the 16 bit matmul Y = X x W

    Key parameters:
    - X: Matrix X with shape (M, K).
    - W: Matrix W with shape (N, K).

    Returns:
    - Y: The output matrix with shape (M, N).
    """
    
    assert A.shape == (8192, 65536), f"Expected A shape (8192, 65536), got {A.shape}"
    assert B.shape == (65536, 28672), f"Expected B shape (65536, 28672), got {B.shape}"

    M, K = x.shape
    K, N = w.shape

    y = torch.empty((M, N), dtype=dtype, device=x.device)

    BLOCK_SIZE_M = 256
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 4
    waves_per_eu = 2
    kpack = 1
    matrix_instr_nonkdim = 16
    num_warps = 8
    num_stages = 2

    # Process blocks directly using .view()
    for i in range(2):
        for j in range(7):
            # Extract A block using .view(): (8192, 65536) -> (2, 4096, 65536)[i]
            x_block = x.view(2, 4096, 65536)[i]  # Shape: (4096, 65536)

            # Extract B block using .view(): (65536, 28672) -> (65536, 7, 4096)[:, j, :]
            w_block = w.view(65536, 7, 4096)[:, j, :].contiguous()  # Shape: (65536, 4096)

            # Create output view directly in final result tensor
            # Map (i,j) block to correct position: (8192, 28672) -> (2, 4096, 7, 4096)[i, :, j, :]
            output_view = y.view(2, 4096, 7, 4096)[i, :, j, :].contiguous()

            grid = lambda META: (triton.cdiv(4096, META['BLOCK_SIZE_M']) * triton.cdiv(4096, META['BLOCK_SIZE_N']), )
            _gemm_a16_w16_kernel[grid](
                x_block,
                w_block,
                output_view,
                M,
                N,
                K,
                x.stride(0),
                x.stride(1),
                w.stride(0),
                w.stride(1),
                y.stride(0),
                y.stride(1),
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                GROUP_SIZE_M,
                waves_per_eu=waves_per_eu,
                kpack=kpack,
                matrix_instr_nonkdim=matrix_instr_nonkdim,
                num_warps=num_warps,
                num_stages=num_stages,
            )

    return y
