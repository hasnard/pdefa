/* ---------------- Conv2d ---------------- */
    case OpType::Conv2d: {
        if (n_ins < 2) return false;
        const Tensor* input  = ins_ptr[0];
        const Tensor* weight = ins_ptr[1];
        const Tensor* bias   = (n_ins > 2) ? ins_ptr[2] : nullptr;

        /* OTOMATİK NHWC DÜZELTME (DFL conv ve benzeri için) */
        Tensor input_fixed;
        if (input->rank() == 4 && weight->rank() == 4 &&
            input->dtype() == DType::F32) {
            const int64_t Cin_w = weight->dim(1);
            if (input->dim(3) != Cin_w && input->dim(2) == Cin_w) {
                const int64_t N = input->dim(0);
                const int64_t H = input->dim(1);
                const int64_t W = input->dim(2);
                const int64_t C = input->dim(3);

                int64_t t_dims[4] = { N, H, C, W };
                Shape t_shape(t_dims, 4);
                input_fixed = Tensor::empty(t_shape, DType::F32);
                if (input_fixed.is_empty()) return false;

                const float* src = input->data<float>();
                float*       dst = input_fixed.data<float>();

                for (int64_t n = 0; n < N; ++n)
                for (int64_t h = 0; h < H; ++h)
                for (int64_t w = 0; w < W; ++w)
                for (int64_t c = 0; c < C; ++c) {
                    const int64_t src_i = ((n*H + h)*W + w)*C + c;
                    const int64_t dst_i = ((n*H + h)*C + c)*W + w;
                    dst[dst_i] = src[src_i];
                }
                input = &input_fixed;
            }
        }

        ops::Conv2dConfig cfg;
        cfg.stride_h    = node.params.stride_h;
        cfg.stride_w    = node.params.stride_w;
        cfg.pad_h       = node.params.pad_h;
        cfg.pad_w       = node.params.pad_w;
        cfg.dil_h       = node.params.dil_h;
        cfg.dil_w       = node.params.dil_w;
        cfg.groups      = (node.params.groups > 0) ? node.params.groups : 1;
        cfg.activation  = static_cast<ops::Conv2dConfig::FusedActivation>(node.params.fused_activation);
        cfg.leaky_slope = node.params.leaky_slope;

        const Shape os = ops::conv2d_output_shape(input->shape(), weight->shape(), cfg);
        if (os.rank() == 0) return false;

        Tensor out = acquire_output_buffer(os, DType::F32);
        if (out.is_empty()) return false;

        ops::conv2d(*input, *weight, bias, out, cfg);



        const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];


        if (out_idx >= 0) intermediate_[static_cast<size_t>(out_idx)] = std::move(out);
        return true;
    }

    /* ---------------- MatMul ---------------- */
    case OpType::MatMul: {
        if (n_ins < 2) return false;
        const Tensor* input  = ins_ptr[0];
        const Tensor* weight = ins_ptr[1];
        const Tensor* bias   = (n_ins > 2) ? ins_ptr[2] : nullptr;

        /* 4D batch matmul (attention, motor NHWC-transposed form):
         *    A: [B, M, K, H]
         *    B: [B, K, N, H]
         *    C: [B, M, N, H] */
        if (input->rank() == 4 && weight->rank() == 4) {
            const int64_t B_ = input->dim(0);
            const int64_t M  = input->dim(1);
            const int64_t K  = input->dim(2);
            const int64_t H_ = input->dim(3);

            if (weight->dim(0) == B_ && weight->dim(1) == K &&
                weight->dim(3) == H_) {
                const int64_t N = weight->dim(2);

                int64_t out_dims[4] = { B_, M, N, H_ };
                Shape os(out_dims, 4);
                Tensor out = acquire_output_buffer(os, DType::F32);
                if (out.is_empty()) return false;

                const float* ap = input->data<float>();
                const float* bp = weight->data<float>();
                float*       cp = out.data<float>();

                for (int64_t b = 0; b < B_; ++b) {
                    for (int64_t h = 0; h < H_; ++h) {
                        for (int64_t i = 0; i < M; ++i) {
                            for (int64_t j = 0; j < N; ++j) {
                                float sum = 0.0f;
                                for (int64_t k = 0; k < K; ++k) {
                                    const float a_v = ap[((b*M + i)*K + k)*H_ + h];
                                    const float b_v = bp[((b*K + k)*N + j)*H_ + h];
                                    sum += a_v * b_v;
                                }
                                cp[((b*M + i)*N + j)*H_ + h] = sum;
                            }
                        }
                    }
                }

                const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
                if (out_idx >= 0) intermediate_[static_cast<size_t>(out_idx)] = std::move(out);
                return true;
            }
        }

                /* ============================================================
         *  4D×2D SON-EKSEN LINEAR (ConvNeXt / ViT MLP)
         *  a=[N,H,W,C], b=[C,Cout] → out=[N,H,W,Cout]
         * ============================================================ */
        if (input->rank() == 4 && weight->rank() == 2 &&
            input->dim(3) == weight->dim(0)) {
            const int64_t N  = input->dim(0);
            const int64_t H  = input->dim(1);
            const int64_t W  = input->dim(2);
            const int64_t C  = input->dim(3);
            const int64_t Co = weight->dim(1);

            int64_t out_dims[4] = { N, H, W, Co };
            Shape os(out_dims, 4);
            Tensor out = acquire_output_buffer(os, DType::F32);
            if (out.is_empty()) return false;

            const float* ap = input->data<float>();
            const float* bp = weight->data<float>();
            const float* bias_ptr = (bias && !bias->is_empty() &&
                                     bias->dtype() == DType::F32 &&
                                     bias->numel() == Co)
                                    ? bias->data<float>() : nullptr;
            float* cp = out.data<float>();

            const int64_t rows = N * H * W;
            auto& pool = runtime::GlobalPool::get();
            constexpr int64_t TILE_ROWS = 64;
            const int64_t n_tiles = (rows + TILE_ROWS - 1) / TILE_ROWS;

            pool.parallel_for(0, n_tiles, [&](int64_t t0, int64_t t1) {
                for (int64_t t = t0; t < t1; ++t) {
                    const int64_t r0 = t * TILE_ROWS;
                    const int64_t r1 = std::min<int64_t>(rows, r0 + TILE_ROWS);
                    for (int64_t r = r0; r < r1; ++r) {
                        const float* a_row = ap + r * C;
                        float*       c_row = cp + r * Co;
                        int64_t co = 0;
                        for (; co + 7 < Co; co += 8) {
                            __m256 acc = bias_ptr
                                ? _mm256_loadu_ps(bias_ptr + co)
                                : _mm256_setzero_ps();
                            for (int64_t ci = 0; ci < C; ++ci) {
                                const __m256 a = _mm256_set1_ps(a_row[ci]);
                                const __m256 b = _mm256_loadu_ps(bp + ci * Co + co);
                                acc = _mm256_fmadd_ps(a, b, acc);
                            }
                            _mm256_storeu_ps(c_row + co, acc);
                        }
                        for (; co < Co; ++co) {
                            float acc = bias_ptr ? bias_ptr[co] : 0.0f;
                            for (int64_t ci = 0; ci < C; ++ci) {
                                acc += a_row[ci] * bp[ci * Co + co];
                            }
                            c_row[co] = acc;
                        }
                    }
                }
            });

            const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
            if (out_idx >= 0) intermediate_[static_cast<size_t>(out_idx)] = std::move(out);
            return true;
        }

        /* 2D / 3D yol (mevcut davranış) */
        const Shape ms = ops::matmul_output_shape(input->shape(), weight->shape());
        if (ms.rank() == 0) return false;

        Tensor out = acquire_output_buffer(ms, DType::F32);
        if (out.is_empty()) return false;

        ops::matmul(*input, *weight, out);

        if (bias && !bias->is_empty() && bias->dtype() == DType::F32) {
            const int64_t M  = out.dim(0);
            const int64_t Nn = out.dim(1);
            float*       op  = out.data<float>();
            const float* bp  = bias->data<float>();
            for (int64_t i = 0; i < M; ++i) {
                float* row = op + i * Nn;
                for (int64_t j = 0; j < Nn; ++j) row[j] += bp[j];
            }
        }

        const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
        if (out_idx >= 0) intermediate_[static_cast<size_t>(out_idx)] = std::move(out);
        return true;
    }

    /* ---------------- Aktivasyonlar ---------------- */
    case OpType::ReLU:
    case OpType::LeakyReLU:
    case OpType::Sigmoid:
    case OpType::Tanh:
    case OpType::SiLU:
    case OpType::GELU:
    case OpType::HardSwish:
    case OpType::HardSigmoid: {
        if (n_ins == 0) return false;

        const Tensor& in = *ins_ptr[0];
        Tensor out = acquire_output_buffer(in.shape(), in.dtype());
        if (out.is_empty()) return false;
        std::memcpy(out.data(), in.data(), in.nbytes());

        switch (node.op) {
            case OpType::ReLU:      ops::relu_(out); break;
            case OpType::LeakyReLU: ops::leaky_relu_(out, node.params.leaky_slope); break;
            case OpType::Sigmoid:   ops::sigmoid_(out); break;
            case OpType::Tanh:      ops::tanh_(out); break;
            case OpType::SiLU:      ops::silu_(out); break;
            case OpType::GELU:      ops::gelu_(out); break;

            case OpType::HardSigmoid: {
                float* p = out.data<float>();
                const int64_t n = out.numel();
                for (int64_t i = 0; i < n; ++i) {
                    float v = p[i] * (1.0f/6.0f) + 0.5f;
                    if (v < 0.0f) v = 0.0f;
                    if (v > 1.0f) v = 1.0f;
                    p[i] = v;
                }
                break;
            }

            case OpType::HardSwish: {
                float* p = out.data<float>();
                const int64_t n = out.numel();
                for (int64_t i = 0; i < n; ++i) {
                    float x = p[i];
                    float v = x * (1.0f/6.0f) + 0.5f;
                    if (v < 0.0f) v = 0.0f;
                    if (v > 1.0f) v = 1.0f;
                    p[i] = x * v;
                }
                break;
            }

            default: break;
        }

        const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
        if (out_idx >= 0) intermediate_[static_cast<size_t>(out_idx)] = std::move(out);
        return true;
    }

    /* ---------------- Pooling ---------------- */
    case OpType::MaxPool2d:
    case OpType::AvgPool2d: {
        if (n_ins == 0) return false;
        ops::Pool2dConfig cfg;
        cfg.kernel_h = node.params.kernel_h;
        cfg.kernel_w = node.params.kernel_w;
        cfg.stride_h = node.params.stride_h;
        cfg.stride_w = node.params.stride_w;
        cfg.pad_h    = node.params.pad_h;
        cfg.pad_w    = node.params.pad_w;

        /* GLOBAL POOLING (ReduceMean / GlobalAveragePool)
         * Input NHWC: [N, H, W, C]  ->  Output NHWC: [N, 1, 1, C] */
        if (cfg.kernel_h <= 0 || cfg.kernel_w <= 0) {
            const Shape& in_s = ins_ptr[0]->shape();
            if (in_s.rank() != 4) return false;

            const int64_t N = in_s.dim(0);
            const int64_t H = in_s.dim(1);
            const int64_t W = in_s.dim(2);
            const int64_t C = in_s.dim(3);
            const int64_t spatial = H * W;
            if (spatial <= 0) return false;

            int64_t out_dims[4] = {N, 1, 1, C};
            Shape os(out_dims, 4);
            Tensor out = acquire_output_buffer(os, DType::F32);
            if (out.is_empty()) return false;

            const float* ip = ins_ptr[0]->data<float>();
            float*       op = out.data<float>();

            auto& pool = runtime::GlobalPool::get();

            if (node.op == OpType::AvgPool2d) {
                pool.parallel_for(0, N * C, [&](int64_t t0, int64_t t1) {
                    for (int64_t t = t0; t < t1; ++t) {
                        const int64_t n = t / C;
                        const int64_t c = t % C;
                        float sum = 0.0f;
                        const float* base = ip + (size_t)n * (size_t)spatial * (size_t)C + (size_t)c;
                        for (int64_t i = 0; i < spatial; ++i) {
                            sum += base[(size_t)i * (size_t)C];
                        }
                        op[(size_t)n * (size_t)C + (size_t)c] = sum / (float)spatial;
                    }
                });
            } else {
                pool.parallel_for(0, N * C, [&](int64_t t0, int64_t t1) {
                    for (int64_t t = t0; t < t1; ++t) {
                        const int64_t n = t / C;
                        const int64_t c = t % C;
                        float mx = -3.402823466e+38f;
                        const float* base = ip + (size_t)n * (size_t)spatial * (size_t)C + (size_t)c;
                        for (int64_t i = 0; i < spatial; ++i) {
                            const float v = base[(size_t)i * (size_t)C];
                            if (v > mx) mx = v;
                        }
                        op[(size_t)n * (size_t)C + (size_t)c] = mx;
                    }
                });
            }

            const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
            if (out_idx >= 0) intermediate_[static_cast<size_t>(out_idx)] = std::move(out);
            return true;
        }

        const Shape os = ops::maxpool2d_output_shape(ins_ptr[0]->shape(), cfg);
        if (os.rank() == 0) return false;

        Tensor out = acquire_output_buffer(os, DType::F32);
        if (out.is_empty()) return false;

        if (node.op == OpType::MaxPool2d) ops::maxpool2d(*ins_ptr[0], out, cfg);
        else                              ops::avgpool2d(*ins_ptr[0], out, cfg);

        const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
        if (out_idx >= 0) intermediate_[static_cast<size_t>(out_idx)] = std::move(out);
        return true;
    }
 

        /* ---------------- LayerNorm (NHWC motor düzeni) ----------------
     * ONNX LayerNormalization:
     *   input[0] = X
     *   input[1] = Scale (gamma)
     *   input[2] = Bias  (beta)
     *   attr axis, epsilon
     * Motor NHWC tuttuğu için son eksen kanal (C). ConvNeXt/ViT/Swin
     * LayerNorm'ları son eksende normalize eder. */
    case OpType::LayerNorm: {
        if (n_ins < 2) return false;
        const Tensor& in = *ins_ptr[0];
        const Tensor& gamma = *ins_ptr[1];
        const Tensor* beta = (n_ins > 2) ? ins_ptr[2] : nullptr;

        const int32_t rank = in.rank();
        int32_t ax = node.params.axis;
        if (ax < 0) ax += rank;
        if (ax < 0 || ax >= rank) return false;

        /* Sadece son eksende normalize destekliyoruz (motor NHWC kanal = son) */
        if (ax != rank - 1) return false;

        const int64_t C = in.dim(rank - 1);
        const int64_t outer = (C > 0) ? (in.numel() / C) : 0;
        const float eps = node.params.eps;

        Tensor out = acquire_output_buffer(in.shape(), DType::F32);
        if (out.is_empty()) return false;

        const float* ip = in.data<float>();
        float*       op_ = out.data<float>();
        const float* g = gamma.data<float>();
        const float* b = (beta && !beta->is_empty() && beta->dtype() == DType::F32)
                         ? beta->data<float>() : nullptr;

        auto& pool = runtime::GlobalPool::get();
        pool.parallel_for(0, outer, [&](int64_t o0, int64_t o1) {
            for (int64_t o = o0; o < o1; ++o) {
                const float* row  = ip + o * C;
                float*       orow = op_ + o * C;

                /* mean */
                float sum = 0.0f;
                for (int64_t c = 0; c < C; ++c) sum += row[c];
                const float mean = sum / (float)C;

                /* variance */
                float var = 0.0f;
                for (int64_t c = 0; c < C; ++c) {
                    const float d = row[c] - mean;
                    var += d * d;
                }
                var /= (float)C;

                const float inv = 1.0f / std::sqrt(var + eps);

                /* normalize + affine */
                if (b) {
                    for (int64_t c = 0; c < C; ++c) {
                        orow[c] = (row[c] - mean) * inv * g[c] + b[c];
                    }
                } else {
                    for (int64_t c = 0; c < C; ++c) {
                        orow[c] = (row[c] - mean) * inv * g[c];
                    }
                }
            }
        });

        const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
        if (out_idx >= 0) intermediate_[static_cast<size_t>(out_idx)] = std::move(out);
        return true;
    }
    


    /* ---------------- BatchNorm2d (NHWC) ---------------- */
    case OpType::BatchNorm2d: {
        if (n_ins < 5) return false;
        const Tensor* input = ins_ptr[0];
        const Tensor& gamma = *ins_ptr[1];
        const Tensor& beta  = *ins_ptr[2];
        const Tensor& mean  = *ins_ptr[3];
        const Tensor& var   = *ins_ptr[4];
        const float eps = node.params.eps;

        /* Motor NHWC: [N,H,W,C], kanal dim 3. NCHW varsayımı yanlış. */
        if (input->rank() == 4 && input->dtype() == DType::F32) {
            const int64_t N = input->dim(0);
            const int64_t H = input->dim(1);
            const int64_t W = input->dim(2);
            const int64_t C = input->dim(3);

            Tensor out = acquire_output_buffer(input->shape(), DType::F32);
            if (out.is_empty()) return false;

            const float* ip = input->data<float>();
            float*       op = out.data<float>();
            const float* g  = gamma.data<float>();
            const float* b  = beta.data<float>();
            const float* m  = mean.data<float>();
            const float* v  = var.data<float>();

            const int64_t HW = H * W;
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t hw = 0; hw < HW; ++hw) {
                    const int64_t base = (n * HW + hw) * C;
                    for (int64_t c = 0; c < C; ++c) {
                        const float scale = g[c] / std::sqrt(v[c] + eps);
                        op[base + c] = (ip[base + c] - m[c]) * scale + b[c];
                    }
                }
            }

            const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
            if (out_idx >= 0) intermediate_[static_cast<size_t>(out_idx)] = std::move(out);
            return true;
        }

        /* Fallback: 4D dışı → eski NCHW yolu */
        Tensor out = acquire_output_buffer(input->shape(), input->dtype());
        if (out.is_empty()) return false;
        std::memcpy(out.data(), input->data(), input->nbytes());
        ops::batchnorm2d_(out, *ins_ptr[1], *ins_ptr[2], *ins_ptr[3], *ins_ptr[4], node.params.eps);

        const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
        if (out_idx >= 0) intermediate_[static_cast<size_t>(out_idx)] = std::move(out);
        return true;
    }
















    /* ---------------- Clip (ReLU6, genel clamp) ---------------- */
    case OpType::Clip: {
        if (n_ins < 1) return false;
        const Tensor& in = *ins_ptr[0];
        Tensor out = acquire_output_buffer(in.shape(), in.dtype());
        if (out.is_empty()) return false;
        std::memcpy(out.data(), in.data(), in.nbytes());

        /* Format-safe: OpParams'ın mevcut float alanlarını yeniden yorumla.
         * Converter Clip min/max'ı buraya yazar. */
        const float lo = node.params.eps;
        const float hi = node.params.leaky_slope;

        float* p = out.data<float>();
        const int64_t n = out.numel();

        const __m256 vlo = _mm256_set1_ps(lo);
        const __m256 vhi = _mm256_set1_ps(hi);
        int64_t i = 0;
        for (; i + 7 < n; i += 8) {
            __m256 v = _mm256_loadu_ps(p + i);
            v = _mm256_max_ps(v, vlo);
            v = _mm256_min_ps(v, vhi);
            _mm256_storeu_ps(p + i, v);
        }
        for (; i < n; ++i) {
            if (p[i] < lo) p[i] = lo;
            if (p[i] > hi) p[i] = hi;
        }

        const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
        if (out_idx >= 0) intermediate_[static_cast<size_t>(out_idx)] = std::move(out);
        return true;
    }