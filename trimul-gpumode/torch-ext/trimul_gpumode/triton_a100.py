import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Set PyTorch flags for performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

@triton.jit
def fused_ln_dual_matmul_kernel(
    # Pointers (9)
    X_ptr, W_4way_ptr, W_og_ptr, Mask_ptr, Norm_Weight_ptr, Norm_Bias_ptr,
    OutLeft_ptr, OutRight_ptr, OutOG_ptr,
    # Metadata (5)
    M, H, K, s1, s2,
    # Strides (16)
    stride_x_m, stride_x_k,
    stride_w4_k, stride_w4_n,
    stride_wog_k, stride_wog_n,
    stride_ol_bs, stride_ol_h, stride_ol_s1, stride_ol_s2,
    stride_or_t_bs, stride_or_t_h, stride_or_t_s2, stride_or_t_s1,
    stride_og_m, stride_og_h,
    stride_mask_m, stride_mask_h,
    # Constexpr (now passed as arguments from the host)
    LN_EPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, H_CHUNK_SIZE: tl.constexpr,
):
    # --- PID Mapping: Based on the LARGER 4*H problem ---
    pid = tl.program_id(axis=0)
    N_4way = 4 * H
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N_4way, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # --- SHARED LayerNorm calculation (done only ONCE) ---
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    m_mask = offs_m < M
    x_rows_base_ptr = X_ptr + offs_m[:, None] * stride_x_m

    mean = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for k_offset in range(0, K, BLOCK_SIZE_K):
        k_chunk_offs = tl.arange(0, BLOCK_SIZE_K)
        x_ptrs = x_rows_base_ptr + (k_offset + k_chunk_offs)[None, :]
        k_mask = (k_offset + k_chunk_offs) < K
        x_chunk = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        mean += tl.sum(x_chunk, axis=1)
    mean /= K

    var = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for k_offset in range(0, K, BLOCK_SIZE_K):
        k_chunk_offs = tl.arange(0, BLOCK_SIZE_K)
        x_ptrs = x_rows_base_ptr + (k_offset + k_chunk_offs)[None, :]
        k_mask = (k_offset + k_chunk_offs) < K
        x_chunk = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        x_centered = x_chunk - mean[:, None]
        var += tl.sum(x_centered * x_centered, axis=1)
    var /= K
    rstd = 1.0 / tl.sqrt(var + LN_EPS)

    # --- Matmul Loop 1: For the 4-Way Projections ---
    offs_n_4way = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    w_4way_ptrs_base = W_4way_ptr + (offs_n_4way[None, :] * stride_w4_n)
    accumulator_4way = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator_og = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    offs_n_og = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_block_start = k * BLOCK_SIZE_K;
        x_ptrs = x_rows_base_ptr + (k_block_start + offs_k)[None, :] * stride_x_k
        w_ptrs = w_4way_ptrs_base + (k_block_start + offs_k)[:, None] * stride_w4_k
        x_mask = (offs_m[:, None] < M) & ((k_block_start + offs_k)[None, :] < K)
        w_mask = ((k_block_start + offs_k)[:, None] < K) & (offs_n_4way[None, :] < N_4way)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)
        norm_w_ptrs = Norm_Weight_ptr + k_block_start + offs_k
        norm_b_ptrs = Norm_Bias_ptr + k_block_start + offs_k
        nw = tl.load(norm_w_ptrs, mask=(k_block_start + offs_k) < K, other=0.0)
        nb = tl.load(norm_b_ptrs, mask=(k_block_start + offs_k) < K, other=0.0)
        x_norm_tile = (x_tile - mean[:, None]) * rstd[:, None]
        x_norm_tile = (x_norm_tile * nw[None, :] + nb[None, :]).to(tl.float16)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        accumulator_4way += tl.dot(x_norm_tile, w_tile)

        if pid_n * BLOCK_SIZE_N < H:
            w_og_ptrs_base = W_og_ptr + (offs_n_og[None, :] * stride_wog_n)
            w_ptrs = w_og_ptrs_base + (k_block_start + offs_k)[:, None] * stride_wog_k
            w_mask = ((k_block_start + offs_k)[:, None] < K) & (offs_n_og[None, :] < H);
            w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
            accumulator_og += tl.dot(x_norm_tile, w_tile)
    
    if pid_n * BLOCK_SIZE_N < H:
        og_out = tl.sigmoid(accumulator_og)
        outg_ptrs = OutOG_ptr + offs_m[:, None] * stride_og_m + offs_n_og[None, :] * stride_og_h
        og_mask = m_mask[:, None] & (offs_n_og[None, :] < H)
        tl.store(outg_ptrs, og_out, mask=og_mask)

    # --- Fusion Logic for 4-Way Part ---
    acc_reshaped = tl.reshape(accumulator_4way, (BLOCK_SIZE_M, H_CHUNK_SIZE, 4))
    role_idx = tl.arange(0, 4)[None, None, :]
    left_proj  = tl.sum(tl.where(role_idx == 0, acc_reshaped, 0.0), axis=2)
    left_gate  = tl.sum(tl.where(role_idx == 1, acc_reshaped, 0.0), axis=2)
    right_proj = tl.sum(tl.where(role_idx == 2, acc_reshaped, 0.0), axis=2)
    right_gate = tl.sum(tl.where(role_idx == 3, acc_reshaped, 0.0), axis=2)
    
    offs_h_chunk = (pid_n * H_CHUNK_SIZE) + tl.arange(0, H_CHUNK_SIZE)
    mask_ptrs = Mask_ptr + offs_m[:, None] * stride_mask_m + offs_h_chunk[None, :] * stride_mask_h
    m_mask_h = m_mask[:, None] & (offs_h_chunk[None, :] < H)
    mask_tile = tl.load(mask_ptrs, mask=m_mask_h, other=0.0)

    left_out = left_proj * tl.sigmoid(left_gate) * mask_tile
    right_out = right_proj * tl.sigmoid(right_gate) * mask_tile

    s1s2 = s1 * s2
    offs_b  = offs_m // s1s2
    offs_s1 = (offs_m % s1s2) // s2
    offs_s2 = offs_m % s2
    offs_b_2d  = tl.reshape(offs_b,  (BLOCK_SIZE_M, 1))
    offs_h_2d  = tl.reshape(offs_h_chunk, (1, H_CHUNK_SIZE))
    offs_s1_2d = tl.reshape(offs_s1, (BLOCK_SIZE_M, 1))
    offs_s2_2d = tl.reshape(offs_s2, (BLOCK_SIZE_M, 1))

    outl_ptrs = OutLeft_ptr + (offs_b_2d * stride_ol_bs + offs_h_2d * stride_ol_h +
                                     offs_s1_2d * stride_ol_s1 + offs_s2_2d * stride_ol_s2)
    outr_ptrs_t = OutRight_ptr + (offs_b_2d * stride_or_t_bs + offs_h_2d * stride_or_t_h +
                                          offs_s2_2d * stride_or_t_s2 + offs_s1_2d * stride_or_t_s1)
    tl.store(outl_ptrs, left_out, mask=m_mask_h)
    tl.store(outr_ptrs_t, right_out, mask=m_mask_h)

@triton.jit
def bmm_coalesced_kernel(
    # Pointers
    Left_ptr, Right_ptr, Out_ptr,
    # Dimensions
    bs, s1, s2, H,
    # Strides
    stride_l_bs, stride_l_h, stride_l_s1, stride_l_s2,
    stride_r_bs, stride_r_h, stride_r_s2, stride_r_s1,
    stride_o_bs, stride_o_h, stride_o_s1, stride_o_s2,
    # Kernel parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(s1, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(s1, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    pid_bh = tl.program_id(axis=1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    left_ptrs_base = Left_ptr + pid_b * stride_l_bs + pid_h * stride_l_h
    right_ptrs_base = Right_ptr + pid_b * stride_r_bs + pid_h * stride_r_h
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(s2, BLOCK_SIZE_K)):
        k_start = k * BLOCK_SIZE_K
        a_ptrs = left_ptrs_base + (offs_m[:, None] * stride_l_s1 + (k_start + offs_k[None, :]) * stride_l_s2)
        b_ptrs = right_ptrs_base + ((k_start + offs_k[:, None]) * stride_r_s2 + offs_n[None, :] * stride_r_s1)
        a_mask = (offs_m[:, None] < s1) & ((k_start + offs_k[None, :]) < s2)
        b_mask = ((k_start + offs_k[:, None]) < s2) & (offs_n[None, :] < s1)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator += tl.dot(a, b)

    out_ptrs = Out_ptr + pid_b * stride_o_bs + pid_h * stride_o_h + \
               offs_m[:, None] * stride_o_s1 + offs_n[None, :] * stride_o_s2
    c_mask = (offs_m[:, None] < s1) & (offs_n[None, :] < s1)
    tl.store(out_ptrs, accumulator, mask=c_mask)

@triton.jit
def fused_final_kernel(
    # Pointers
    In_ptr, Gate_ptr, NormW_ptr, NormB_ptr, ProjW_ptr, Out_ptr,
    # Metadata
    M, H, D, s1,
    # Strides
    stride_in_bs, stride_in_h, stride_in_s1_row, stride_in_s1_col,
    stride_gate_m, stride_gate_h,
    stride_proj_d, stride_proj_h,
    stride_out_bs, stride_out_s1_row, stride_out_s1_col, stride_out_d,
    # Constants
    LN_EPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(D, BLOCK_SIZE_N)
    
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    m_mask = offs_m < M

    s1s1 = s1 * s1
    b = offs_m // s1s1
    r = (offs_m % s1s1) // s1
    c = offs_m % s1

    sum_x = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    sum_x2 = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    in_ptr_base = In_ptr + b * stride_in_bs + r * stride_in_s1_row + c * stride_in_s1_col

    for k_offset in range(0, H, BLOCK_SIZE_K):
        offs_k = k_offset + tl.arange(0, BLOCK_SIZE_K)
        k_mask = offs_k < H
        in_ptrs = in_ptr_base[:, None] + offs_k[None, :] * stride_in_h
        in_chunk = tl.load(in_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        sum_x += tl.sum(in_chunk, axis=1)
        sum_x2 += tl.sum(in_chunk * in_chunk, axis=1)
        
    mean = sum_x / H
    var = (sum_x2 / H) - (mean * mean)
    rstd = tl.math.rsqrt(var + LN_EPS)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_offset in range(0, H, BLOCK_SIZE_K):
        offs_k = k_offset + tl.arange(0, BLOCK_SIZE_K)
        k_mask = offs_k < H
        in_ptrs = in_ptr_base[:, None] + offs_k[None, :] * stride_in_h
        a = tl.load(in_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        a_norm = (a - mean[:, None]) * rstd[:, None]
        norm_w = tl.load(NormW_ptr + offs_k, mask=k_mask, other=0.0)
        norm_b = tl.load(NormB_ptr + offs_k, mask=k_mask, other=0.0)
        a_norm = a_norm * norm_w[None, :] + norm_b[None, :]
        proj_ptrs = ProjW_ptr + offs_n[None, :] * stride_proj_d + offs_k[:, None] * stride_proj_h
        gate_ptrs = Gate_ptr + offs_m[:, None] * stride_gate_m + offs_k[None, :] * stride_gate_h
        gate = tl.load(gate_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        a_gated = a_norm * gate
        b_w = tl.load(proj_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < D), other=0.0)
        acc += tl.dot(a_gated.to(b_w.dtype), b_w)
        
    offs_d = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptr_base = Out_ptr + b*stride_out_bs + r*stride_out_s1_row + c*stride_out_s1_col
    out_ptrs = out_ptr_base[:, None] + offs_d[None, :] * stride_out_d
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & (offs_d[None, :] < D))

def compiledtrimul_fused_interleaved_final(
    x: torch.Tensor,
    mask_mh: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    W_4way: torch.Tensor,
    W_og: torch.Tensor,
    to_out_norm_weight: torch.Tensor,
    to_out_norm_bias: torch.Tensor,
    to_out_weight: torch.Tensor,
    h: int,
):
    bs, s1, s2, d = x.shape
    M, K, H = bs * s1 * s2, x.shape[-1], h
    x_flat = x.view(M, K)

    left_final  = torch.empty((bs, H, s1, s2), device=x.device, dtype=torch.float16)
    right_final_t = torch.empty((bs, H, s2, s1), device=x.device, dtype=torch.float16)
    og_mh = torch.empty((M, H), device=x.device, dtype=torch.float16)

    # --- Kernel 1: Fused LN + Dual Matmul ---
    N_4way = 4 * H
    # Hardcoded A100 best config: M128-N128-K32-GM8-HC32-W8-S2
    config_k1 = {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'H_CHUNK_SIZE': 32}
    grid_k1 = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N_4way, meta['BLOCK_SIZE_N']),)
    
    fused_ln_dual_matmul_kernel[grid_k1](
        x_flat, W_4way, W_og, mask_mh, norm_weight, norm_bias,
        left_final, right_final_t, og_mh,
        M, H, K, s1, s2,
        x_flat.stride(0), x_flat.stride(1), W_4way.stride(0), W_4way.stride(1),
        W_og.stride(0), W_og.stride(1), left_final.stride(0), left_final.stride(1),
        left_final.stride(2), left_final.stride(3), right_final_t.stride(0), right_final_t.stride(1),
        right_final_t.stride(2), right_final_t.stride(3), og_mh.stride(0), og_mh.stride(1),
        mask_mh.stride(0), mask_mh.stride(1),
        LN_EPS=1e-5, **config_k1, num_warps=8, num_stages=2
    )
    
    # --- Kernel 2: Batched Matrix Multiplication ---
    bmm_out_tmp = torch.empty((bs, H, s1, s1), device=x.device, dtype=torch.float16)
    # Hardcoded A100 best config: M128-N64-K32-GM8-W4-S3
    config_k2 = {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}
    grid_k2 = lambda meta: (triton.cdiv(s1, meta['BLOCK_SIZE_M']) * triton.cdiv(s1, meta['BLOCK_SIZE_N']), bs * H)
    
    bmm_coalesced_kernel[grid_k2](
        left_final, right_final_t, bmm_out_tmp,
        bs, s1, s2, H,
        left_final.stride(0), left_final.stride(1), left_final.stride(2), left_final.stride(3),
        right_final_t.stride(0), right_final_t.stride(1), right_final_t.stride(2), right_final_t.stride(3),
        bmm_out_tmp.stride(0), bmm_out_tmp.stride(1), bmm_out_tmp.stride(2), bmm_out_tmp.stride(3),
        **config_k2, num_warps=4, num_stages=3
    )

    # --- Kernel 3: Fully Fused Final Stage ---
    final_out = torch.empty((bs, s1, s1, d), device=x.device, dtype=torch.float16)
    # Hardcoded A100 best config: M32-N128-K32-GM8-W4-S3
    config_k3 = {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}
    grid_k3 = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(d, meta['BLOCK_SIZE_N']),)
    
    fused_final_kernel[grid_k3](
        bmm_out_tmp, og_mh, to_out_norm_weight, to_out_norm_bias, to_out_weight, final_out,
        M, H, d, s1,
        bmm_out_tmp.stride(0), bmm_out_tmp.stride(1), bmm_out_tmp.stride(2), bmm_out_tmp.stride(3),
        og_mh.stride(0), og_mh.stride(1), to_out_weight.stride(0), to_out_weight.stride(1),
        final_out.stride(0), final_out.stride(1), final_out.stride(2), final_out.stride(3),
        LN_EPS=1e-5, **config_k3, num_warps=4, num_stages=3
    )
    return final_out

def pack_w_4way_efficient(weights):
    """ Packs L, LG, R, RG into a tight [K, 4*H] matrix. """
    WL, WLG, WR, WRG = (weights[k] for k in ['left_proj.weight', 'left_gate.weight', 'right_proj.weight', 'right_gate.weight'])
    H, K = WL.shape
    ws = torch.stack([WL, WLG, WR, WRG], dim=0).permute(1, 0, 2).contiguous().view(4 * H, K)
    return ws.t().to(torch.float16)

def get_w_og(weights):
    """ Gets the transposed [K, H] out_gate weight matrix. """
    return weights['out_gate.weight'].t().to(torch.float16)

@torch.compile()
def compiledtrimul(
    x: torch.Tensor, mask: torch.Tensor, norm_weight: torch.Tensor, norm_bias: torch.Tensor,
    w_concat: torch.Tensor, to_out_norm_weight: torch.Tensor, to_out_norm_bias: torch.Tensor,
    to_out_weight: torch.Tensor, h: int
) -> torch.Tensor:
    bs, s1, s2, d = x.shape
    x_norm = F.layer_norm(x, (d,), norm_weight, norm_bias).view((bs * s1 * s2, d)).to(torch.float16)
    all_projections = torch.mm(x_norm, w_concat)
    left, right, lg, rg, og = all_projections.chunk(5, dim=1)
    mask_expanded = mask.expand(-1, -1, -1, h).reshape(-1, h)
    left = left * mask_expanded * torch.sigmoid(lg)
    right = right * mask_expanded * torch.sigmoid(rg)
    out_gate = torch.sigmoid(og)
    left = left.view(bs, s1, s2, h).permute(0,3,1,2)
    right = right.view(bs, s1, s2, h).permute(0,3,1,2)
    out_p = torch.matmul(left.to(torch.float16), right.to(torch.float16).transpose(-1, -2))
    out_einsum_flat = out_p.permute(0,2,3,1).reshape(bs * s1 * s1, h)
    normed = F.layer_norm(out_einsum_flat, (h,), to_out_norm_weight, to_out_norm_bias).to(torch.float16)
    gated = normed * out_gate
    final_out_flat = gated @ to_out_weight.t()
    return final_out_flat.view(bs, s1, s1, d)

def small_kernel_pt_path(data):
    input_tensor, mask, weights, config = data
    w_concat = torch.cat([
        weights['left_proj.weight'], weights['right_proj.weight'], weights['left_gate.weight'],
        weights['right_gate.weight'], weights['out_gate.weight']
    ], dim=0).t().contiguous().to(torch.float16)
    return compiledtrimul(
        x=input_tensor.to(torch.float32), mask=mask.unsqueeze(-1),
        norm_weight=weights['norm.weight'].to(torch.float32),
        norm_bias=weights['norm.bias'].to(torch.float32), w_concat=w_concat,
        to_out_norm_weight=weights['to_out_norm.weight'].to(torch.float16),
        to_out_norm_bias=weights['to_out_norm.bias'].to(torch.float16),
        to_out_weight=weights['to_out.weight'].to(torch.float16),
        h=config["hidden_dim"]
    )

def kernel_a100(data):
    input_tensor, mask, weights, config = data
    bs, s1, s2, d = input_tensor.shape
    
    if s1 < 512: # Adjusted threshold based on observed BMM configs
        return small_kernel_pt_path(data)

    H = config["hidden_dim"]
    W_4way = pack_w_4way_efficient(weights)
    W_og = get_w_og(weights)
    M = bs * s1 * s2
    mask_mh = mask.unsqueeze(-1).expand(-1, -1, -1, H).reshape(M, H).to(torch.float16)

    return compiledtrimul_fused_interleaved_final(
        x=input_tensor.to(torch.float32),
        mask_mh=mask_mh,
        norm_weight=weights['norm.weight'].to(torch.float32),
        norm_bias=weights['norm.bias'].to(torch.float32),
        W_4way=W_4way,
        W_og=W_og,
        to_out_norm_weight=weights['to_out_norm.weight'].to(torch.float16),
        to_out_norm_bias=weights['to_out_norm.bias'].to(torch.float16),
        to_out_weight=weights['to_out.weight'].to(torch.float16),
        h=H,
    )