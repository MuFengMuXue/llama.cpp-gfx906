// Arithmetic intensity probe for quantized MUL_MAT on a ggml backend.
//
// Runs the same K x N matmul with Q4_0, Q8_0 and IQ4_NL weights at batch 1
// (decode) and at a prompt batch (prefill), and prints, per type, the bytes
// the kernel must move, the FLOPs it must retire, and the measured time,
// effective bandwidth and throughput.
//
// On gfx906 the fork also has a repacked-weight path for Q8_0, Q4_0 and IQ4_NL.
// Those rows are measured too (disable with --no-repack) and the repacked
// output is compared against the canonical one (nmse column).
//
// Build: see run.sh

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static const char * type_name(ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q4_0:   return "q4_0";
        case GGML_TYPE_Q8_0:   return "q8_0";
        case GGML_TYPE_IQ4_NL: return "iq4_nl";
        default:               return ggml_type_name(t);
    }
}

struct result {
    double time_s = 0.0;
    double gbps   = 0.0;
    double tflops = 0.0;
};

// One matmul, weights of type `type`, activations f32. The graph holds `repeat`
// independent copies of the matmul to keep launch overhead out of the timing.
// If `w_buft` is non-null the weights are allocated and uploaded through that
// buffer type (the gfx906 repack path); otherwise the backend default is used.
static result bench_type(ggml_backend_t backend, ggml_backend_buffer_type_t w_buft,
                         ggml_type type, int64_t K, int64_t N, int64_t n_tokens,
                         const std::vector<float> & src, int iters, int repeat,
                         std::vector<float> * out_copy) {
    ggml_init_params params = {
        /* .mem_size   = */ 8ull*1024*1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    // Canonical: one context, so the weight tensor is followed by the
    // activations and the outputs in the same buffer: some kernels over-read
    // the last block a few bytes past its end, which a weight-only buffer
    // would turn into a fault.
    // Repacked: the weight lives in the repack buffer type, the activations
    // and outputs in the default one (the repack buffer cannot read back).
    // A dummy tensor follows the weight so the repack buffer is not sized
    // exactly to the last row.
    ggml_context * ctx_w = nullptr;
    ggml_tensor * w = nullptr;
    if (w_buft != nullptr) {
        ctx_w = ggml_init(params);
        w = ggml_new_tensor_2d(ctx_w, type, K, N);
        ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, 4096);
    }

    ggml_context * ctx = ggml_init(params);
    if (w_buft == nullptr) {
        w = ggml_new_tensor_2d(ctx, type, K, N);
    }
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, n_tokens);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * out0 = nullptr;
    for (int r = 0; r < repeat; ++r) {
        ggml_tensor * out = ggml_mul_mat(ctx, w, x);
        if (r == 0) {
            out0 = out;
        }
        ggml_build_forward_expand(gf, out);
    }

    ggml_backend_buffer_t buf_w = w_buft != nullptr
        ? ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, w_buft)
        : nullptr;
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr || (w_buft != nullptr && buf_w == nullptr)) {
        fprintf(stderr, "%s: failed to allocate backend buffer\n", type_name(type));
        exit(1);
    }
    std::vector<uint8_t> q(ggml_nbytes(w));
    ggml_quantize_chunk(type, src.data(), q.data(), 0, N, K, nullptr);
    ggml_backend_tensor_set(w, q.data(), 0, q.size());

    std::vector<float> xv(K*n_tokens);
    for (int64_t i = 0; i < K*n_tokens; ++i) {
        xv[i] = std::sin(0.001f*i);
    }
    ggml_backend_tensor_set(x, xv.data(), 0, xv.size()*sizeof(float));

    for (int i = 0; i < 3; ++i) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);

    const double t0 = now_s();
    for (int i = 0; i < iters; ++i) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);
    const double t1 = now_s();

    result r;
    r.time_s = (t1 - t0)/iters/repeat;

    const double wbytes = (double) ggml_nbytes(w);
    const double abytes = (double) (K + N)*n_tokens*sizeof(float);
    const double flops  = 2.0*K*N*n_tokens;
    r.gbps   = (wbytes + abytes)/r.time_s/1e9;
    r.tflops = flops/r.time_s/1e12;

    if (out_copy != nullptr) {
        out_copy->resize((size_t) N*n_tokens);
        ggml_backend_tensor_get(out0, out_copy->data(), 0, out_copy->size()*sizeof(float));
    }

    // kept alive until exit: the backend's q8_1 activation cache keys on tensor
    // and data pointers, and reusing those addresses with a freed entry would
    // hand a stale buffer to the next run
    (void) buf;
    (void) buf_w;
    (void) ctx;
    (void) ctx_w;
    return r;
}

static ggml_backend_buffer_type_t find_repack_buft(ggml_backend_dev_t dev) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto get_extra = (ggml_backend_dev_get_extra_bufts_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    if (get_extra == nullptr) {
        return nullptr;
    }
    ggml_backend_buffer_type_t * bufts = get_extra(dev);
    for (int i = 0; bufts != nullptr && bufts[i] != nullptr; ++i) {
        if (strstr(ggml_backend_buft_name(bufts[i]), "Repack") != nullptr) {
            return bufts[i];
        }
    }
    return nullptr;
}

static bool repack_supports(ggml_type type) {
    return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_Q4_0;
}

int main(int argc, char ** argv) {
    int64_t K = 8192;
    int64_t N = 8192;
    bool use_repack = true;
    int  only_layout = -1; // -1: both, 0: canonical, 1: repacked
    const char * only_type = nullptr;
    int  repeat_override = 0; // 0: auto
    std::vector<int64_t> batches = { 1, 128 };
    std::string backend_dir = "build/bin";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--backend-dir") == 0 && i + 1 < argc) {
            backend_dir = argv[++i];
        } else if (strcmp(argv[i], "-K") == 0 && i + 1 < argc) {
            K = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-N") == 0 && i + 1 < argc) {
            N = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--no-repack") == 0) {
            use_repack = false;
        } else if (strcmp(argv[i], "--canonical-only") == 0) {
            only_layout = 0;
        } else if (strcmp(argv[i], "--repacked-only") == 0) {
            only_layout = 1;
        } else if (strcmp(argv[i], "--only") == 0 && i + 1 < argc) {
            only_type = argv[++i];
        } else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
            repeat_override = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            batches.clear();
            const char * p = argv[++i];
            while (*p != '\0') {
                char * end = nullptr;
                batches.push_back(strtoll(p, &end, 10));
                if (end == p) {
                    break;
                }
                p = *end == ',' ? end + 1 : end;
            }
        }
    }

    ggml_backend_load_all_from_path(backend_dir.c_str());
    ggml_backend_load_all();

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev == nullptr) {
        fprintf(stderr, "no GPU backend found (backend dir: %s)\n", backend_dir.c_str());
        return 1;
    }

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) {
        fprintf(stderr, "failed to init GPU backend\n");
        return 1;
    }

    ggml_backend_buffer_type_t repack_buft = use_repack ? find_repack_buft(dev) : nullptr;

    size_t free_mem = 0, total_mem = 0;
    ggml_backend_dev_memory(dev, &free_mem, &total_mem);
    printf("device      : %s (%s)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
    printf("vram        : %.1f / %.1f GiB free\n", free_mem/1024.0/1024/1024, total_mem/1024.0/1024/1024);
    printf("repack buft : %s\n", repack_buft != nullptr ? ggml_backend_buft_name(repack_buft) : "none");
    printf("matmul      : K=%lld N=%lld (weights streamed once per call)\n\n", (long long) K, (long long) N);

    std::vector<float> src((size_t) K*N);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = std::sin(0.0007f*i);
    }

    const ggml_type types[] = { GGML_TYPE_Q4_0, GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_0 };

    printf("%-7s %-10s %5s %8s %8s %8s %9s %8s %8s %10s\n",
           "type", "layout", "bpw", "AI(w)", "AI(all)", "tokens", "time_us", "GB/s", "TFLOP/s", "nmse");
    printf("------------------------------------------------------------------------------------------------\n");

    for (int64_t n_tokens : batches) {
        const int iters   = n_tokens == 1 ? 50 : 10;
        const int repeat  = repeat_override > 0 ? repeat_override : (n_tokens == 1 ? 16 : 2);
        for (ggml_type type : types) {
            if (only_type != nullptr && strcmp(only_type, type_name(type)) != 0) {
                continue;
            }
            std::vector<float> ref;
            // canonical layout first, then the repacked one when the type has it
            for (int layout = 0; layout < 2; ++layout) {
                const bool repacked = layout == 1;
                if (only_layout >= 0 && layout != only_layout) {
                    continue;
                }
                if (repacked && (repack_buft == nullptr || !repack_supports(type))) {
                    continue;
                }
                std::vector<float> outc;
                result r = bench_type(backend, repacked ? repack_buft : nullptr, type,
                                      K, N, n_tokens, src, iters, repeat, &outc);

                double nmse = -1.0;
                if (repacked) {
                    double err = 0.0, norm = 0.0;
                    for (size_t i = 0; i < ref.size(); ++i) {
                        const double d = (double) outc[i] - (double) ref[i];
                        err  += d*d;
                        norm += (double) ref[i]*ref[i];
                    }
                    nmse = norm > 0.0 ? err/norm : 0.0;
                } else {
                    ref = std::move(outc);
                }

                const double bpw    = 8.0*ggml_type_size(type)/ggml_blck_size(type);
                const double wb     = (double) K*N*ggml_type_size(type)/ggml_blck_size(type);
                const double ab     = (double) (K + N)*n_tokens*sizeof(float);
                const double flops  = 2.0*K*N*n_tokens;

                printf("%-7s %-10s %5.2f %8.2f %8.2f %8lld %9.1f %8.1f %8.3f ",
                       type_name(type), repacked ? "repacked" : "canonical", bpw,
                       flops/wb, flops/(wb + ab), (long long) n_tokens,
                       r.time_s*1e6, r.gbps, r.tflops);
                if (repacked) {
                    printf("%10.2e\n", nmse);
                } else {
                    printf("%10s\n", "-");
                }
            }
        }
        printf("\n");
    }

    ggml_backend_free(backend);
    ggml_quantize_free();
    return 0;
}
