// MoE entry point (mul_mat_id) for repacked Q8_0 / Q4_0 / IQ4_NL weights.
#include "repack.cuh"
#include "repack-common.cuh"
#include "repack-kernels.cuh"
#include "quantize.cuh"
#include "mmq.cuh"
#include "mmid.cuh"

// Upstream's ids helper sorts the assignments by expert but does not return the
// per-expert counts the GEMM tiles need, so build the prefix sum here.
static __global__ void repack_expert_bounds(const int32_t * __restrict__ ids,
        int32_t * __restrict__ bounds, const int n_expert, const int n_tokens,
        const int n_expert_used, const int si1) {
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) {
        bounds[e + 1] = 0;
    }
    __syncthreads();
    const int n_assign = n_tokens * n_expert_used;
    for (int i = threadIdx.x; i < n_assign; i += blockDim.x) {
        const int e = ids[(i / n_expert_used) * si1 + (i % n_expert_used)];
        atomicAdd(&bounds[e + 1], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        bounds[0] = 0;
        for (int e = 0; e < n_expert; e++) {
            bounds[e + 1] += bounds[e];
        }
    }
}

void ggml_cuda_mul_mat_id_repacked(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ids->type  == GGML_TYPE_I32);
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(ids->nb[0]  == sizeof(int32_t));
    GGML_ASSERT(src1->ne[3] == 1 && dst->ne[3] == 1);
    GGML_ASSERT(src1->nb[2] == src1->nb[1] * src1->ne[1]);
    GGML_ASSERT(dst->nb[2]  == dst->nb[1]  * dst->ne[1]);
    GGML_ASSERT(dst->nb[1]  == (size_t) dst->ne[0] * sizeof(float));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 == ne00);
    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens      = ids->ne[1];
    const int64_t n_assign      = n_expert_used * n_tokens;

    cudaStream_t stream = ctx.stream();

    const uint8_t * w;
    if (src0->view_src != nullptr && ggml_cuda_repack_tensor_supported(src0->view_src)) {
        w = repack_view_get_cached(src0, src0->view_src, stream);
    } else {
        w = (const uint8_t *) src0->data;
    }
    float * dst_d = (float *) dst->data;
    const size_t expert_stride = repack_gcn_nbytes(src0->type, ne00, ne01);
    const uint32_t dst_s1 = dst->nb[1] / sizeof(float);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_assign);
    ggml_cuda_pool_alloc<int32_t> ids_dst (ctx.pool(), n_assign);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);
    // upstream's helper always writes its tile spec; the repack GEMM builds its
    // own tiles, so the buffer is only here to keep the helper happy.
    ggml_cuda_pool_alloc<ggml_cuda_mm_id_tile_spec> tile_spec(ctx.pool(), (size_t) n_assign + ne02);
    {
        const int si1  = ids->nb[1] / sizeof(int32_t);
        const int sis1 = src1->nb[2] / src1->nb[1];
        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data,
            ids_src1.get(), ids_dst.get(), tile_spec.get(),
            ne02, n_tokens, n_expert_used, src1->ne[1], /*tile_width =*/ 1, si1, sis1, false, stream);
        repack_expert_bounds<<<1, 256, 0, stream>>>(
            (const int32_t *) ids->data, expert_bounds.get(), ne02, n_tokens, n_expert_used, si1);
        CUDA_CHECK(cudaGetLastError());
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const int64_t x_stride    = ne10_padded / QK8_1;
    const int64_t n_cols      = src1->ne[1] * src1->ne[2];
    const int64_t s11         = src1->nb[1] / sizeof(float);

    if (n_tokens <= MMQ_RP_Q8_MOE_MMV_MAX_TOKENS) {
        ggml_cuda_pool_alloc<block_q8_1> xq_own;
        block_q8_1 * xq = repack_quantize_src1_q8_1(ctx, src0, src1, ne10,
            ne10_padded, s11, s11 * n_cols, s11 * n_cols, n_cols, 1, 1,
            (size_t) (n_cols * x_stride), xq_own, stream);

        const dim3 grid((ne01 + 3) / 4, (unsigned) n_assign, 1);
        switch (src0->type) {
            case GGML_TYPE_Q8_0:
                mul_mat_vec_repacked_id1<4, 2, 2><<<grid, 128, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    ids_src1.get(), ids_dst.get(), expert_bounds.get(), (uint32_t) ne02,
                    expert_stride, (uint32_t) x_stride, dst_s1);
                break;
            case GGML_TYPE_Q4_0:
                mul_mat_vec_repacked_id1<4, 2, 2, GGML_TYPE_Q4_0><<<grid, 128, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    ids_src1.get(), ids_dst.get(), expert_bounds.get(), (uint32_t) ne02,
                    expert_stride, (uint32_t) x_stride, dst_s1);
                break;
            case GGML_TYPE_IQ4_NL:
                mul_mat_vec_repacked_id1<4, 2, 2, GGML_TYPE_IQ4_NL><<<grid, 128, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    ids_src1.get(), ids_dst.get(), expert_bounds.get(), (uint32_t) ne02,
                    expert_stride, (uint32_t) x_stride, dst_s1);
                break;
            default:
                GGML_ABORT("unsupported repack type");
        }
        return;
    }

    const uint64_t n_groups = (uint64_t) ne10_padded / (4 * QK8_1);
    ggml_cuda_pool_alloc<char> xq_buf(ctx.pool(),
        (size_t) n_cols * n_groups * sizeof(block_q8_1_mmq_h));
    quantize_mmq_q8_1_cuda((const float *) src1->data, nullptr, xq_buf.get(),
        repack_mmq_x_type(src0->type), ne10, s11, s11 * n_cols, s11 * n_cols, ne10_padded,
        n_cols, 1, 1, stream);
    const block_q8_1 * xq = (const block_q8_1 *) xq_buf.get();

    constexpr int BN_ID  = 64 * MMQ_RP_Q8_TN;
    constexpr int BN_W32 = 32;
    const bool use_w32 = n_assign < 2 * BN_ID * ne02;

    switch (src0->type) {
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_IQ4_NL: {
            if (use_w32) {
                const int64_t max_tiles = n_assign / BN_W32 + ne02;
                ggml_cuda_pool_alloc<int32_t>          tile_off (ctx.pool(), ne02 + 1);
                ggml_cuda_pool_alloc<repack_tile_meta> tile_meta(ctx.pool(), max_tiles);
                repack_tile_off<BN_W32><<<1, 1, 0, stream>>>(expert_bounds.get(), tile_off.get(), tile_meta.get(), ne02);
                const dim3 grid((ne01 + MMQ_RP_Q8_BM - 1) / MMQ_RP_Q8_BM, max_tiles, 1);
                switch (src0->type) {
                    case GGML_TYPE_Q8_0:
                        mmq_gemm_repacked_w32<true, 1, MMQ_RP_Q8_NROW_LANES * 2, GGML_TYPE_Q8_0><<<grid, dim3(32, MMQ_RP_Q8_NROW_LANES * 2), 0, stream>>>(
                            w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) n_cols,
                            ids_src1.get(), ids_dst.get(), expert_bounds.get(), tile_off.get(), tile_meta.get(),
                            (uint32_t) ne02, expert_stride, dst_s1);
                        break;
                    case GGML_TYPE_Q4_0:
                        mmq_gemm_repacked_w32<true, 1, MMQ_RP_Q8_NROW_LANES * 2, GGML_TYPE_Q4_0><<<grid, dim3(32, MMQ_RP_Q8_NROW_LANES * 2), 0, stream>>>(
                            w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) n_cols,
                            ids_src1.get(), ids_dst.get(), expert_bounds.get(), tile_off.get(), tile_meta.get(),
                            (uint32_t) ne02, expert_stride, dst_s1);
                        break;
                    default:
                        mmq_gemm_repacked_w32<true, 1, MMQ_RP_Q8_NROW_LANES * 2, GGML_TYPE_IQ4_NL><<<grid, dim3(32, MMQ_RP_Q8_NROW_LANES * 2), 0, stream>>>(
                            w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) n_cols,
                            ids_src1.get(), ids_dst.get(), expert_bounds.get(), tile_off.get(), tile_meta.get(),
                            (uint32_t) ne02, expert_stride, dst_s1);
                        break;
                }
            } else {
                const int64_t max_tiles = n_assign / BN_ID + ne02;
                ggml_cuda_pool_alloc<int32_t>          tile_off (ctx.pool(), ne02 + 1);
                ggml_cuda_pool_alloc<repack_tile_meta> tile_meta(ctx.pool(), max_tiles);
                repack_tile_off<BN_ID><<<1, 1, 0, stream>>>(expert_bounds.get(), tile_off.get(), tile_meta.get(), ne02);
                const dim3 grid((ne01 + MMQ_RP_Q8_BM - 1) / MMQ_RP_Q8_BM, max_tiles, 1);
                switch (src0->type) {
                    case GGML_TYPE_Q8_0:
                        mmq_gemm_repacked<true, MMQ_RP_Q8_TN, MMQ_RP_Q8_NROW_LANES, GGML_TYPE_Q8_0><<<grid, dim3(64, MMQ_RP_Q8_NROW_LANES), 0, stream>>>(
                            w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) n_cols,
                            ids_src1.get(), ids_dst.get(), expert_bounds.get(), tile_off.get(), tile_meta.get(),
                            (uint32_t) ne02, expert_stride, dst_s1);
                        break;
                    case GGML_TYPE_Q4_0:
                        mmq_gemm_repacked<true, MMQ_RP_Q8_TN, MMQ_RP_Q8_NROW_LANES, GGML_TYPE_Q4_0><<<grid, dim3(64, MMQ_RP_Q8_NROW_LANES), 0, stream>>>(
                            w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) n_cols,
                            ids_src1.get(), ids_dst.get(), expert_bounds.get(), tile_off.get(), tile_meta.get(),
                            (uint32_t) ne02, expert_stride, dst_s1);
                        break;
                    default:
                        mmq_gemm_repacked<true, MMQ_RP_Q8_TN, MMQ_RP_Q8_NROW_LANES, GGML_TYPE_IQ4_NL><<<grid, dim3(64, MMQ_RP_Q8_NROW_LANES), 0, stream>>>(
                            w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) n_cols,
                            ids_src1.get(), ids_dst.get(), expert_bounds.get(), tile_off.get(), tile_meta.get(),
                            (uint32_t) ne02, expert_stride, dst_s1);
                        break;
                }
            }
        } break;
        default:
            GGML_ABORT("unsupported repack type");
    }
}
