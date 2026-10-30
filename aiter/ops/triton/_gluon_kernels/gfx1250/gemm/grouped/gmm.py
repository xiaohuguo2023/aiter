# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Persistent grouped GEMM (GMM) for gfx1250, using TDM descriptor loads.

Ported from the TLX reference kernel in
``third_party/tlx/tutorials/amd_grouped_gemm_gfx1250`` of the TLX Triton fork.
This is the core schedule only: a persistent program walks the tiles of every
group in turn and runs a multi-buffered TDM pipeline over K. The reference's
optional modes (cluster multicast, cross-tile prefetch, C staging variants,
XCD remapping) are not ported yet.

Unlike the reference, which takes ``B`` as ``[G, N, K]`` and transposes each
tile in LDS, this kernel takes the non-transposed ``(G, K, N)`` layout that
``gmm`` already uses, so the B tile loads straight into the WMMA B operand.

TDM descriptors carry the real extents, and ``async_load`` with explicit
offsets clamps to them, so partial M tiles and a ragged K tail need no
hand-written masks.
"""

import triton.experimental.gluon.language as gl
import triton.language as tl
from triton.experimental import gluon

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_gmm_gluon_repr = make_kernel_repr(
    "_gmm_gluon_kernel",
    ["BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M", "NUM_BUFFERS", "NUM_PROGRAMS"],
)


@gluon.jit
def _tile_offsets(
    tile_idx_in_group,
    num_m_tiles,
    num_n_tiles,
    GROUP_M: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    """Group-ordered (m, n) tile offsets, so neighbouring tiles share B tiles."""
    num_pid_in_group = GROUP_M * num_n_tiles
    group_id = tile_idx_in_group // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_m_tiles - first_pid_m, GROUP_M)
    tile_m = first_pid_m + ((tile_idx_in_group % num_pid_in_group) % group_size_m)
    tile_n = (tile_idx_in_group % num_pid_in_group) // group_size_m
    return tile_m * BLOCK_M, tile_n * BLOCK_N


@gluon.jit(repr=_gmm_gluon_repr)
def _gmm_gluon_kernel(
    A,
    B,
    C,
    GroupSizes,
    Bias,
    N,
    K,
    G,
    stride_am,
    stride_ak,
    stride_bg,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias_g,
    NUM_PROGRAMS: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GROUP_M: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    USE_BIAS: gl.constexpr,
    num_warps: gl.constexpr,
):
    NUM_TDM_OPS: gl.constexpr = 2  # A, B
    gl.static_assert(NUM_BUFFERS >= 2, "NUM_BUFFERS must be at least 2")
    gl.static_assert(num_warps == 4 or num_warps == 8, "num_warps must be 4 or 8")
    # TDM encodes the pad interval in dwords in a 3-bit log2 field, so it cannot
    # exceed 256 dwords, i.e. 512 16-bit elements.
    gl.static_assert(BLOCK_K <= 512, "BLOCK_K exceeds the TDM pad interval")
    gl.static_assert(BLOCK_N <= 512, "BLOCK_N exceeds the TDM pad interval")

    if num_warps == 4:
        WARP_BASES: gl.constexpr = [[0, 1], [0, 2]]
    else:
        WARP_BASES: gl.constexpr = [[0, 1], [0, 2], [1, 0]]

    WMMA_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=WARP_BASES,
        reg_bases=[],
        instr_shape=[16, 16, 32],
    )
    DOT_LAYOUT_A: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=WMMA_LAYOUT, k_width=8
    )
    DOT_LAYOUT_B: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=WMMA_LAYOUT, k_width=8
    )

    SHARED_LAYOUT_A: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_K, 8]], [BLOCK_M, BLOCK_K], [1, 0]
    )
    # B is (K, N): already the WMMA B-operand orientation, so no transpose.
    SHARED_LAYOUT_B: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_N, 16]], [BLOCK_K, BLOCK_N], [1, 0]
    )
    SHARED_LAYOUT_C: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_N, 8]], [BLOCK_M, BLOCK_N], [1, 0]
    )

    a_buffer = gl.allocate_shared_memory(
        A.dtype.element_ty, [NUM_BUFFERS, BLOCK_M, BLOCK_K], SHARED_LAYOUT_A
    )
    b_buffer = gl.allocate_shared_memory(
        B.dtype.element_ty, [NUM_BUFFERS, BLOCK_K, BLOCK_N], SHARED_LAYOUT_B
    )
    c_buffer = gl.allocate_shared_memory(
        C.dtype.element_ty, [BLOCK_M, BLOCK_N], SHARED_LAYOUT_C
    )

    num_n_tiles = tl.cdiv(N, BLOCK_N)
    num_k_iter = tl.cdiv(K, BLOCK_K)

    tile_idx = gl.program_id(0)
    last_problem_end = 0
    m_start = 0

    for g in range(G):
        # group_sizes may be int64, but a TDM descriptor's shape must be 32-bit.
        # Only the base pointer offsets below are widened.
        gm = gl.load(GroupSizes + g).to(gl.int32)
        num_m_tiles = tl.cdiv(gm, BLOCK_M)
        num_tiles = num_m_tiles * num_n_tiles

        # Walk every tile of this group that belongs to this persistent program.
        while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            off_m, off_n = _tile_offsets(
                tile_idx - last_problem_end,
                num_m_tiles,
                num_n_tiles,
                GROUP_M,
                BLOCK_M,
                BLOCK_N,
            )

            # Descriptors are rebuilt per tile: the K cursor restarts at 0 and
            # the group base moves with m_start / g.
            a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                base=A + m_start.to(gl.int64) * stride_am,
                shape=(gm, K),
                strides=(stride_am, stride_ak),
                block_shape=(BLOCK_M, BLOCK_K),
                layout=SHARED_LAYOUT_A,
            )
            b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                base=B + g.to(gl.int64) * stride_bg,
                shape=(K, N),
                strides=(stride_bk, stride_bn),
                block_shape=(BLOCK_K, BLOCK_N),
                layout=SHARED_LAYOUT_B,
            )

            write_idx = 0
            read_idx = 0

            for _ in gl.static_range(NUM_BUFFERS - 1):
                w = write_idx % NUM_BUFFERS
                gl.amd.gfx1250.tdm.async_load(
                    a_desc, [off_m, write_idx * BLOCK_K], a_buffer.index(w)
                )
                gl.amd.gfx1250.tdm.async_load(
                    b_desc, [write_idx * BLOCK_K, off_n], b_buffer.index(w)
                )
                write_idx += 1

            acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)

            for _ in range(num_k_iter - (NUM_BUFFERS - 1)):
                w = write_idx % NUM_BUFFERS
                gl.amd.gfx1250.tdm.async_load(
                    a_desc, [off_m, write_idx * BLOCK_K], a_buffer.index(w)
                )
                gl.amd.gfx1250.tdm.async_load(
                    b_desc, [write_idx * BLOCK_K, off_n], b_buffer.index(w)
                )
                write_idx += 1

                gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * NUM_TDM_OPS)
                r = read_idx % NUM_BUFFERS
                a_tile = a_buffer.index(r).load(layout=DOT_LAYOUT_A)
                b_tile = b_buffer.index(r).load(layout=DOT_LAYOUT_B)
                acc = gl.amd.gfx1250.wmma(a_tile, b_tile, acc)
                read_idx += 1

            for i in gl.static_range(NUM_BUFFERS - 1):
                gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * NUM_TDM_OPS)
                r = read_idx % NUM_BUFFERS
                a_tile = a_buffer.index(r).load(layout=DOT_LAYOUT_A)
                b_tile = b_buffer.index(r).load(layout=DOT_LAYOUT_B)
                acc = gl.amd.gfx1250.wmma(a_tile, b_tile, acc)
                read_idx += 1

            if USE_BIAS:
                bias_offs = off_n + gl.arange(
                    0, BLOCK_N, layout=gl.SliceLayout(0, WMMA_LAYOUT)
                )
                bias = gl.amd.cdna3.buffer_load(
                    Bias + g.to(gl.int64) * stride_bias_g,
                    bias_offs,
                    mask=bias_offs < N,
                    other=0.0,
                )
                acc = acc + bias[None, :].to(gl.float32)

            c_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                base=C + m_start.to(gl.int64) * stride_cm,
                shape=(gm, N),
                strides=(stride_cm, stride_cn),
                block_shape=(BLOCK_M, BLOCK_N),
                layout=SHARED_LAYOUT_C,
            )
            c_buffer.store(acc.to(C.dtype.element_ty))
            gl.amd.gfx1250.tdm.async_store(c_desc, [off_m, off_n], c_buffer)
            # The next tile reuses c_buffer, so the store must land first.
            gl.amd.gfx1250.tdm.async_wait(0)

            tile_idx += NUM_PROGRAMS

        last_problem_end += num_tiles
        m_start += gm
