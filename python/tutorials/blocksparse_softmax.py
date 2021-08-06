import os

import pytest
import torch
import triton
import triton.language as tl


def next_power_of_2(n):
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n


def num_warps(n):
    if n < 512:
        return 4
    if n < 2048:
        return 8
    return 16


@triton.heuristics(
    {"num_warps": lambda *args, **meta: num_warps(args[6] * meta["BLOCK_SIZE"])}
)
@triton.heuristics(
    {
        "BLOCK_SIZE_N": lambda *args, **meta: next_power_of_2(
            args[6] * meta["BLOCK_SIZE"]
        )
    }
)
@triton.jit
def softmax_forward_kernel(
    # Pointer to a tensor that is (batch, n_heads, M, N) and we want to compute the
    # softmax across the M dimension
    input_ptr,
    # Scale to apply to the logits before softmax. This can be helpful because the softmax
    # is computed in fp32 which prevents overflow or underflow when scaling
    scale,
    lut_ptr,  # Pointer to some pre-computed information we store
    unused_param_1,  # Deadlocks if this is removed
    key_padding_mask,
    attention_mask,
    unused_param_2,  # Deadlocks if this is removed
    stride_batch_x,
    stride_batch_key_padding_mask,
    stride_batch_attn_mask,
    **meta
):
    BLOCK_SIZE_N = meta["BLOCK_SIZE_N"]
    BLOCK_SIZE_M = meta["BLOCK_SIZE"]
    # We parallelize across both the batch and the row dimensions
    pid_head_row = tl.program_id(0)
    pid_batch = tl.program_id(1)
    # create index ranges
    rxm = pid_head_row % BLOCK_SIZE_M
    rbm = pid_head_row // BLOCK_SIZE_M
    rxn = tl.arange(0, BLOCK_SIZE_N) % BLOCK_SIZE_M
    rbn = tl.arange(0, BLOCK_SIZE_N) // BLOCK_SIZE_M
    # extract information from LUT
    header = lut_ptr + rbm * 2
    size = tl.load(header + 0)
    offset = tl.load(header + 1)
    check = rbn < size
    rbmn = tl.where(check, rbn, size - 1)
    # block id and column id
    blockid = tl.load(lut_ptr + offset + rbmn * 4 + 0)
    columnid = tl.load(lut_ptr + offset + rbmn * 4 + 1)
    rowid = tl.load(lut_ptr + offset + rbmn * 4 + 2)
    # pointers to X
    px = (
        input_ptr
        + pid_batch * stride_batch_x
        + blockid * BLOCK_SIZE_M * BLOCK_SIZE_M
        + rxm * BLOCK_SIZE_M
        + rxn
    )
    x = tl.load(px, mask=check, other=-float("inf"))
    # This was a suggestion from Philippe. We are unsure if it improves performance
    # or not. Seems inconsistent. It may need to go in another place.
    x = tl.multiple_of(x, 8)  # compiler hint
    x = x.to(tl.float32)
    # apply scale
    if meta["APPLY_SCALE"]:
        x = x * scale
    # apply key-padding mask
    if meta["APPLY_KP_MASK"]:
        pkp_m = (
            key_padding_mask
            + pid_batch * stride_batch_key_padding_mask
            + columnid * BLOCK_SIZE_M
            + rxn
        )
        kp_m = tl.load(pkp_m, mask=check, other=-float("inf"))
        if meta["KP_MASK_MUL"]:
            kp_m = tl.where(kp_m == 0, -float("inf"), 0.0)
        x = x + kp_m
    # apply attention mask
    if meta["APPLY_ATTN_MASK"]:
        pattn_m = (
            attention_mask
            + columnid * BLOCK_SIZE_M
            + rowid * BLOCK_SIZE_M * stride_batch_attn_mask
            + rxm * stride_batch_attn_mask
            + rxn
        )
        attn_m = tl.load(pattn_m, mask=check, other=-float("inf"))
        if meta["ATTN_MASK_MUL"]:
            attn_m = tl.where(attn_m == 0, -float("inf"), 0.0)
        x = x + attn_m
    # computation
    x = tl.softmax(x)
    tl.store(px, x, mask=check)


@triton.heuristics(
    {"num_warps": lambda *args, **meta: num_warps(args[4] * meta["block_size"])}
)
@triton.heuristics(
    {"TN": lambda *args, **meta: next_power_of_2(args[4]) * meta["block_size"]}
)
@triton.jit
def softmax_backward_kernel(X, scale, DX, LUT, sizemax, stride_zx, stride_zdx, **meta):
    pidhm = tl.program_id(0)
    pidz = tl.program_id(1)
    TN = meta["TN"]
    block_size = meta["block_size"]
    # create index ranges
    rxm = pidhm % block_size
    rbm = pidhm // block_size
    rxn = tl.arange(0, TN) % block_size
    rbn = tl.arange(0, TN) // block_size
    # extract information from look-up table
    header = LUT + rbm * 2
    size = tl.load(header + 0)
    offset = tl.load(header + 1)
    # bounds checking on lut
    check = rbn < size
    rbmn = tl.where(check, rbn, size - 1)
    # initialize pointers to block_size-sparse input
    block_id = tl.load(LUT + offset + rbmn * 4)
    X = (
        X
        + pidz * stride_zx
        + block_id * block_size * block_size
        + rxm * block_size
        + rxn
    )
    DX = (
        DX
        + pidz * stride_zdx
        + block_id * block_size * block_size
        + rxm * block_size
        + rxn
    )
    # compute fused softmax backward
    x = tl.load(X, mask=check, other=0)
    dx = tl.load(DX, mask=check, other=0)
    x = x.to(tl.float32)
    dx = dx.to(tl.float32)
    y = x * (dx - tl.sum(x * dx, 0)) * scale
    tl.store(DX, y, mask=check)


class BlocksparseSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        scale,
        key_padding_mask,
        attn_mask,
        kp_mask_mode,
        attn_mask_mode,
        spdims,
        block_size,
        lut,
        maxlut,
        bench,
        time,
    ):
        apply_scale = False if scale == 1.0 else True

        # handle None key_padding_mask
        if key_padding_mask is None:
            apply_kp_mask = False
            stride_zkpm = 0
            key_padding_mask = torch.empty(0, dtype=x.dtype, device=x.device)
        else:
            apply_kp_mask = True
            stride_zkpm = key_padding_mask.stride(0)

        # handle None attention_mask
        if attn_mask is None:
            apply_attn_mask = False
            stride_zattnm = 0
            attn_mask = torch.empty(0, dtype=x.dtype, device=x.device)
        else:
            apply_attn_mask = True
            stride_zattnm = attn_mask.stride(0)

        # run kernel
        M = x.shape[0]
        meta = {
            "BLOCK_SIZE": block_size,
            "APPLY_SCALE": apply_scale,
            "APPLY_KP_MASK": apply_kp_mask,
            "APPLY_ATTN_MASK": apply_attn_mask,
            "KP_MASK_MUL": kp_mask_mode == "mul",
            "ATTN_MASK_MUL": attn_mask_mode == "mul",
        }
        grid = lambda opt: [spdims[0] * spdims[1] * block_size, M]
        # softmax_forward[grid](
        #     x, scale, lut,  key_padding_mask, attn_mask, maxlut, x.stride(0),
        #     stride_zkpm, stride_zattnm, force_nc_cache=True, **meta
        # )
        softmax_forward_kernel[grid](
            x,
            scale,
            lut,
            0,
            key_padding_mask,
            attn_mask,
            maxlut,
            x.stride(0),
            stride_zkpm,
            stride_zattnm,
            force_nc_cache=True,
            **meta
        )

        # save to context
        ctx.mark_dirty(x)
        ctx.save_for_backward(x, lut)
        ctx.spdims = spdims
        ctx.block_size = block_size
        ctx.maxlut = maxlut
        ctx.scale = scale
        ctx.apply_scale = apply_scale
        ctx.apply_kp_mask = apply_kp_mask
        ctx.apply_attn_mask = apply_attn_mask
        ctx.kp_mask_mode = kp_mask_mode
        ctx.attn_mask_mode = attn_mask_mode
        return x

    @staticmethod
    def backward(ctx, dx):
        # retrieve from context
        x, lut = ctx.saved_tensors
        # run kernel
        M = x.shape[0]
        grid = lambda opt: [ctx.spdims[0] * ctx.spdims[1] * ctx.block_size, M]
        softmax_backward_kernel[grid](
            x,
            ctx.scale,
            dx,
            lut,
            ctx.maxlut,
            x.stride(0),
            dx.stride(0),
            force_nc_cache=True,
            block_size=ctx.block_size,
        )
        return (
            dx,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class BlocksparseSoftmax:
    def make_lut(self, device):
        key = (device,)
        if key not in self.lut_cache:
            self.lut_cache[key] = make_lut(self.block_mask)
        return self.lut_cache[key]

    def __init__(self, block_mask: torch.Tensor, block_size: int, bench=False):
        self.spdims = block_mask.shape
        self.block_mask = block_mask
        self.block_size = block_size
        self.bench = bench
        self.lut_cache = dict()

    def __call__(
        self,
        x,
        scale=1.0,
        key_padding_mask=None,
        attn_mask=None,
        key_padding_mask_mode="add",
        attn_mask_mode="add",
    ):
        time_y = [None]
        if attn_mask is not None and attn_mask.dtype != x.dtype:
            raise ValueError("Attention mask must be %s" % x.dtype)
        if key_padding_mask is not None and key_padding_mask.dtype != x.dtype:
            raise ValueError("Key padding mask must be %s" % x.dtype)
        print("making the lut")
        lut = self.make_lut(x.device)
        print("made the lut")
        x = BlocksparseSoftmaxFunction.apply(
            x,
            scale,
            key_padding_mask,
            attn_mask,
            key_padding_mask_mode,
            attn_mask_mode,
            self.spdims,
            self.block_size,
            lut,
            1,
            self.bench,
            time_y,
        )
        return x


def make_lut(block_mask: torch.Tensor):
    """Cache some information for lookup in the kernel. This is cached into a flat
    tensor so it can easily be read in Triton.

    Specifically, we cache a few pieces of information:
    1. block_idx: which block we are processing. This is just an arange
    2. indices for the nonzero blocks. These are flattened out to being
        all the column indices then all the row indices then all the head indices.
    3.

    :param block_mask: The block mask. Has shape (n_heads, n_blocks, n_blocks)

    """
    # block indices
    block_idxs = torch.arange(block_mask.sum())
    head, rows, columns = block_mask.nonzero().split(1, dim=1)
    core = torch.stack(
        (block_idxs, columns[:, 0], rows[:, 0], head[:, 0]), dim=1
    ).flatten()
    # construct look-up table

    num_blocks_per_head_and_row = block_mask.sum(dim=-1).flatten()
    block_offsets = torch.zeros_like(num_blocks_per_head_and_row)
    block_offsets[1:] = torch.cumsum(num_blocks_per_head_and_row[:-1], dim=0)
    block_offsets = block_offsets * 4 + 2 * num_blocks_per_head_and_row.numel()
    header = torch.stack((num_blocks_per_head_and_row, block_offsets), dim=1).flatten()
    lut = torch.cat((header, core)).type(torch.int32).cuda()
    return lut


@pytest.mark.parametrize(
    "block_size, n_ctx", [(32, 256), (32, 576), (32, 1024), (32, 1792)]
)
def test_softmax(block_size, n_ctx, dtype=torch.float16):
    # set seed
    torch.random.manual_seed(0)
    batch_size, n_heads, n_ctx_ks, n_ctx_qs = 2, 4, n_ctx, n_ctx
    scale = 0.4
    # this is a block attention mask
    block_mask = torch.randint(
        2, (n_heads, n_ctx_ks // block_size, n_ctx_qs // block_size), dtype=torch.bool
    )
    logits = torch.randn(
        (batch_size, n_heads, n_ctx_ks, n_ctx_qs),
        dtype=dtype,
        requires_grad=True,
        device="cuda",
    )
    fine_mask = torch.randint(
        low=0,
        high=2,
        size=(n_ctx_qs, n_ctx_qs),
        dtype=torch.bool,
        requires_grad=False,
        device="cuda",
    )
    key_padding_mask = torch.randint(
        low=0,
        high=2,
        size=(batch_size, n_ctx_qs),
        dtype=dtype,
        requires_grad=False,
        device="cuda",
    )
    key_padding_mask[key_padding_mask == 1.0] = float("-inf")
    # triton result
    sparse_softmax = BlocksparseSoftmax(block_mask, block_size)
    print("made it")
    triton_inputs = triton.testing.sparsify_tensor(logits, block_mask, block_size)
    print("running it")
    triton_outputs = sparse_softmax(
        triton_inputs,
        scale=scale,
        key_padding_mask=key_padding_mask,
        key_padding_mask_mode="add",
        attn_mask=fine_mask.to(dtype),
        attn_mask_mode="mul",
    )
    print("ran it")
    # torch result
    torch_inputs = triton.testing.mask_tensor(
        logits, block_mask, block_size, value=float("-inf")
    )
    if fine_mask is not None:
        # broadcast fine_mask to the same shape as inputs
        n_ctx_ks = fine_mask[None, None, :, :] + torch.zeros_like(torch_inputs)
        torch_inputs[n_ctx_ks == 0] = float("-inf")
    if key_padding_mask is not None:
        torch_inputs += key_padding_mask[:, None, None, :]
    torch_outputs = torch.softmax(torch_inputs * scale, -1)
    torch_outputs_sparse = triton.testing.sparsify_tensor(
        torch_outputs, block_mask, block_size
    )
    # compare
    assert triton.testing.allclose(torch_outputs_sparse, triton_outputs)


# @triton.heuristics({'num_warps': lambda *args, **meta: num_warps(args[6] * meta['block_size'])})
# @triton.heuristics({'TN': lambda *args, **meta: next_power_of_2(args[6] * meta['block_size'])})
# @triton.jit
# def softmax_forward_kernel(
#     input_ptr, # Ptr to logits to compute the softmax of
#     # Scale to apply to the logits before softmax. This can be helpful because the softmax
#     # is computed in fp32 which prevents overflow or underflow when scaling
#     scale,
#     lut_ptr, # No idea
#     key_padding_mask,
#     attention_mask,
#     # Stride to get to next batch element in x
#     stride_batch_x,
#     stride_batch_key_padding_mask,
#     strize_batch_attn_mask,
#     **meta
# ):
#     TN = meta['TN']
#     block_size = meta['block_size']
#     pidhm = tl.program_id(0)
#     pidz = tl.program_id(1)
#     # create index ranges
#     rxm = pidhm % block_size
#     rbm = pidhm // block_size
#     rxn = tl.arange(0, TN) % block_size
#     rbn = tl.arange(0, TN) // block_size
#     # extract information from LUT
#     header = lut_ptr + rbm * 2
#     size = tl.load(header + 0)
#     offset = tl.load(header + 1)
#     check = rbn < size
#     rbmn = tl.where(check, rbn, size - 1)
#     # block_size id and column id
#     block_id = tl.load(lut_ptr + offset + rbmn * 4 + 0)
#     columnid = tl.load(lut_ptr + offset + rbmn * 4 + 1)
#     rowid = tl.load(lut_ptr + offset + rbmn * 4 + 2)
#     headid = tl.load(lut_ptr + offset + rbmn * 4 + 3)
#     # pointers to X
#     px = input_ptr + pidz * stride_batch_x + block_id * block_size * block_size + rxm * block_size + rxn
#     x = tl.load(px, mask=check, other=-float('inf'))
#     x = x.to(tl.float32)
#     # apply scale
#     if meta['APPLY_SCALE']:
#         x = x * scale
#     # apply key-padding mask
#     if meta['APPLY_KP_MASK']:
#         pkp_m = key_padding_mask + pidz * stride_batch_key_padding_mask + columnid * block_size + rxn
#         kp_m = tl.load(pkp_m, mask=check, other=-float('inf'))
#         if meta['KP_MASK_MUL']:
#             kp_m = tl.where(kp_m == 0, -float('inf'), 0.)
#         x = x + kp_m
#     # apply attention mask
#     if meta['APPLY_ATTN_MASK']:
#         pattn_m = attention_mask + columnid * block_size + rowid * block_size * strize_batch_attn_mask + rxm * strize_batch_attn_mask + rxn
#         attn_m = tl.load(pattn_m, mask=check, other=-float('inf'))
#         if meta['ATTN_MASK_MUL']:
#             attn_m = tl.where(attn_m == 0, -float('inf'), 0.)
#         x = x + attn_m
#     # computation
#     x = tl.softmax(x)
#     tl.store(px, x, mask=check)
