
    
    /* ---------------- Elementwise (Add, Mul, Sub, Div) ---------------- */
    case OpType::Add:
    case OpType::Mul:
    case OpType::Sub:
    case OpType::Div: {
        if (n_ins < 2) return false;

        const Tensor& a = *ins_ptr[0];
        const Tensor& b = *ins_ptr[1];

        const int64_t ra = a.rank(), rb = b.rank();
        const int64_t ro = std::max(ra, rb);
        if (ro > 8) return false;

        int64_t out_dims[8];
        for (int64_t d = 0; d < ro; ++d) {
            const int64_t da = (d < ro - ra) ? 1 : a.dim(d - (ro - ra));
            const int64_t db = (d < ro - rb) ? 1 : b.dim(d - (ro - rb));
            if (da != db && da != 1 && db != 1) return false;
            out_dims[d] = (da > db) ? da : db;
        }
        Shape os(out_dims, (int32_t)ro);

        Tensor out = acquire_output_buffer(os, a.dtype());
        if (out.is_empty()) return false;

        const float* ap = a.data<float>();
        const float* bp = b.data<float>();
        float* op_ = out.data<float>();

        const int64_t no = out.numel();
        const int64_t na = a.numel();
        const int64_t nb = b.numel();
        const int op = (int)node.op;

        auto& pool = runtime::GlobalPool::get();
           





        /* ============================================================
         *  CHANNEL BROADCAST FAST PATH — SE blokları için
         * ============================================================ */
        {
            const int64_t C_last = os.dim(ro - 1);

            bool b_is_channel = (rb >= 1 && b.dim(rb - 1) == C_last);
            if (b_is_channel) {
                for (int64_t d = 0; d < rb - 1; ++d) {
                    if (b.dim(d) != 1) { b_is_channel = false; break; }
                }
            }
            if (b_is_channel && nb == C_last && na == no && a.dim(ra - 1) == C_last) {
                const int64_t outer = no / C_last;
                pool.parallel_for(0, outer, [&](int64_t o0, int64_t o1) {
                    for (int64_t o = o0; o < o1; ++o) {
                        const float* a_row = ap + o * C_last;
                        float*       o_row = op_ + o * C_last;
                        int64_t c = 0;
                        if (op == (int)OpType::Mul) {
                            for (; c + 7 < C_last; c += 8) {
                                _mm256_storeu_ps(o_row + c,
                                    _mm256_mul_ps(_mm256_loadu_ps(a_row + c),
                                                  _mm256_loadu_ps(bp + c)));
                            }
                            for (; c < C_last; ++c) o_row[c] = a_row[c] * bp[c];
                        } else if (op == (int)OpType::Add) {
                            for (; c + 7 < C_last; c += 8) {
                                _mm256_storeu_ps(o_row + c,
                                    _mm256_add_ps(_mm256_loadu_ps(a_row + c),
                                                  _mm256_loadu_ps(bp + c)));
                            }
                            for (; c < C_last; ++c) o_row[c] = a_row[c] + bp[c];
                        } else if (op == (int)OpType::Sub) {
                            for (; c + 7 < C_last; c += 8) {
                                _mm256_storeu_ps(o_row + c,
                                    _mm256_sub_ps(_mm256_loadu_ps(a_row + c),
                                                  _mm256_loadu_ps(bp + c)));
                            }
                            for (; c < C_last; ++c) o_row[c] = a_row[c] - bp[c];
                        } else { /* Div */
                            for (; c + 7 < C_last; c += 8) {
                                _mm256_storeu_ps(o_row + c,
                                    _mm256_div_ps(_mm256_loadu_ps(a_row + c),
                                                  _mm256_loadu_ps(bp + c)));
                            }
                            for (; c < C_last; ++c) o_row[c] = a_row[c] / bp[c];
                        }
                    }
                });
                const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
                if (out_idx >= 0) intermediate_[static_cast<size_t>(out_idx)] = std::move(out);
                return true;
            }

            /* ============================================================
             *  TERS YÖN: a=kanal-only, b=full (ONNX export sırası)
             *  örn. a=[1,1,1,144], b=[1,56,56,144] — SE bloğu scale*x
             * ============================================================ */
            bool a_is_channel = (ra >= 1 && a.dim(ra - 1) == C_last);
            if (a_is_channel) {
                for (int64_t d = 0; d < ra - 1; ++d) {
                    if (a.dim(d) != 1) { a_is_channel = false; break; }
                }
            }
            if (a_is_channel && na == C_last && nb == no && b.dim(rb - 1) == C_last) {
                const int64_t outer = no / C_last;
                pool.parallel_for(0, outer, [&](int64_t o0, int64_t o1) {
                    for (int64_t o = o0; o < o1; ++o) {
                        const float* b_row = bp + o * C_last;
                        float*       o_row = op_ + o * C_last;
                        int64_t c = 0;
                        if (op == (int)OpType::Mul) {
                            for (; c + 7 < C_last; c += 8) {
                                _mm256_storeu_ps(o_row + c,
                                    _mm256_mul_ps(_mm256_loadu_ps(ap + c),
                                                  _mm256_loadu_ps(b_row + c)));
                            }
                            for (; c < C_last; ++c) o_row[c] = ap[c] * b_row[c];
                        } else if (op == (int)OpType::Add) {
                            for (; c + 7 < C_last; c += 8) {
                                _mm256_storeu_ps(o_row + c,
                                    _mm256_add_ps(_mm256_loadu_ps(ap + c),
                                                  _mm256_loadu_ps(b_row + c)));
                            }
                            for (; c < C_last; ++c) o_row[c] = ap[c] + b_row[c];
                        } else if (op == (int)OpType::Sub) {
                            /* a - b: kanal - full */
                            for (; c + 7 < C_last; c += 8) {
                                _mm256_storeu_ps(o_row + c,
                                    _mm256_sub_ps(_mm256_loadu_ps(ap + c),
                                                  _mm256_loadu_ps(b_row + c)));
                            }
                            for (; c < C_last; ++c) o_row[c] = ap[c] - b_row[c];
                        } else { /* Div */
                            for (; c + 7 < C_last; c += 8) {
                                _mm256_storeu_ps(o_row + c,
                                    _mm256_div_ps(_mm256_loadu_ps(ap + c),
                                                  _mm256_loadu_ps(b_row + c)));
                            }
                            for (; c < C_last; ++c) o_row[c] = ap[c] / b_row[c];
                        }
                    }
                });
                const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
                if (out_idx >= 0) intermediate_[static_cast<size_t>(out_idx)] = std::move(out);
                return true;
            }
        }




    

        if (na == no && nb == no) {
            pool.parallel_for(0, no, [&](int64_t start, int64_t end) {
                int64_t i = start;
                if (op == (int)OpType::Add) {
                    for (; i + 7 < end; i += 8) _mm256_storeu_ps(op_ + i, _mm256_add_ps(_mm256_loadu_ps(ap + i), _mm256_loadu_ps(bp + i)));
                    for (; i < end; ++i) op_[i] = ap[i] + bp[i];
                } else if (op == (int)OpType::Mul) {
                    for (; i + 7 < end; i += 8) _mm256_storeu_ps(op_ + i, _mm256_mul_ps(_mm256_loadu_ps(ap + i), _mm256_loadu_ps(bp + i)));
                    for (; i < end; ++i) op_[i] = ap[i] * bp[i];
                } else if (op == (int)OpType::Sub) {
                    for (; i + 7 < end; i += 8) _mm256_storeu_ps(op_ + i, _mm256_sub_ps(_mm256_loadu_ps(ap + i), _mm256_loadu_ps(bp + i)));
                    for (; i < end; ++i) op_[i] = ap[i] - bp[i];
                } else {
                    /* IEEE 754 dogal davranis: 0'a bolme = inf/nan */
                    for (; i < end; ++i) op_[i] = ap[i] / bp[i];
                }
            });
        }
        else if (nb == 1) {
            const float bv0 = bp[0];
            const __m256 b_vec = _mm256_set1_ps(bv0);

            pool.parallel_for(0, no, [&](int64_t start, int64_t end) {
                int64_t i = start;
                if (op == (int)OpType::Add) {
                    for (; i + 7 < end; i += 8) _mm256_storeu_ps(op_ + i, _mm256_add_ps(_mm256_loadu_ps(ap + i), b_vec));
                    for (; i < end; ++i) op_[i] = ap[i] + bv0;
                } else if (op == (int)OpType::Mul) {
                    for (; i + 7 < end; i += 8) _mm256_storeu_ps(op_ + i, _mm256_mul_ps(_mm256_loadu_ps(ap + i), b_vec));
                    for (; i < end; ++i) op_[i] = ap[i] * bv0;
                } else if (op == (int)OpType::Sub) {
                    for (; i + 7 < end; i += 8) _mm256_storeu_ps(op_ + i, _mm256_sub_ps(_mm256_loadu_ps(ap + i), b_vec));
                    for (; i < end; ++i) op_[i] = ap[i] - bv0;
                } else {
                    for (; i < end; ++i) op_[i] = ap[i] / bv0;
                }
            });
        }
        else {
            for (int64_t i = 0; i < no; ++i) {
                int64_t t = i;
                for (int64_t d = ro - 1; d >= 0; --d) {
                    out_dims[d] = t % os.dim(d);
                    t /= os.dim(d);
                }
                int64_t ai = 0, bi = 0;
                for (int64_t d = 0; d < ra; ++d) ai = ai * a.dim(d) + ((a.dim(d) == 1) ? 0 : out_dims[ro - ra + d]);
                for (int64_t d = 0; d < rb; ++d) bi = bi * b.dim(d) + ((b.dim(d) == 1) ? 0 : out_dims[ro - rb + d]);

                const float av = ap[ai], bv = bp[bi];
                if (op == (int)OpType::Add) op_[i] = av + bv;
                else if (op == (int)OpType::Mul) op_[i] = av * bv;
                else if (op == (int)OpType::Sub) op_[i] = av - bv;
                else op_[i] = av / bv;
            }
        }

        const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
        if (out_idx >= 0) intermediate_[static_cast<size_t>(out_idx)] = std::move(out);
        return true;
    }

    /* ---------------- Softmax ---------------- */
            case OpType::Softmax: {
        if (n_ins == 0) return false;
        const Tensor& in = *ins_ptr[0];

        int64_t ax = node.params.axis;
        if (ax < 0) ax += in.rank();

        /* NHWC remap: ONNX NCHW eksenini motor NHWC-transposed eksenine çevir.
         * Debug: motor tensörü [1, 8400, 16, 4] (NHWC-transposed).
         * ONNX axis=1 (16 bins) → motor axis=2 */
        if (in.rank() == 4 && in.dtype() == DType::F32) {
            static const int64_t remap[4] = {0, 2, 3, 1};
            if (ax >= 0 && ax < 4) ax = remap[ax];
        }

        if (ax < 0 || ax >= in.rank()) return false;

        int64_t outer = 1;
        for (int64_t d = 0; d < ax; ++d) outer *= in.dim(d);
        const int64_t axis_dim = in.dim(ax);
        int64_t inner = 1;
        for (int64_t d = ax + 1; d < in.rank(); ++d) inner *= in.dim(d);

        Tensor out = acquire_output_buffer(in.shape(), in.dtype());
        if (out.is_empty()) return false;

        const float* ip = in.data<float>();
        float* op_ = out.data<float>();

        auto& pool = runtime::GlobalPool::get();

        constexpr int64_t K_BLK = 512;
        const int64_t num_k_blks = (inner + K_BLK - 1) / K_BLK;
        const int64_t total_tasks = outer * num_k_blks;

        pool.parallel_for(0, total_tasks, [&](int64_t t0, int64_t t1) {
            for (int64_t t = t0; t < t1; ++t) {
                const int64_t o = t / num_k_blks;
                const int64_t b = t % num_k_blks;
                const int64_t k0 = b * K_BLK;
                const int64_t k_end = std::min<int64_t>(inner, k0 + K_BLK);

                /* Vektörel kısım: 8'li gruplar */
                int64_t kk = k0;
                for (; kk + 8 <= k_end; kk += 8) {
                    const float* base     = ip  + o * axis_dim * inner + kk;
                    float*       out_base = op_ + o * axis_dim * inner + kk;

                    __m256 vmax = _mm256_set1_ps(-1e30f);
                    for (int64_t a = 0; a < axis_dim; ++a) {
                        __m256 v = _mm256_loadu_ps(base + a * inner);
                        vmax = _mm256_max_ps(vmax, v);
                    }

                    __m256 vsum = _mm256_setzero_ps();
                    for (int64_t a = 0; a < axis_dim; ++a) {
                        __m256 v = _mm256_loadu_ps(base + a * inner);
                        alignas(32) float tmp_in[8], tmp_out[8];
                        _mm256_store_ps(tmp_in, _mm256_sub_ps(v, vmax));
                        for (int k = 0; k < 8; ++k) tmp_out[k] = std::exp(tmp_in[k]);
                        __m256 e = _mm256_load_ps(tmp_out);
                        _mm256_storeu_ps(out_base + a * inner, e);
                        vsum = _mm256_add_ps(vsum, e);
                    }

                    __m256 vrecip = softmax_rcp_avx2(vsum);
                    for (int64_t a = 0; a < axis_dim; ++a) {
                        __m256 e = _mm256_loadu_ps(out_base + a * inner);
                        _mm256_storeu_ps(out_base + a * inner, _mm256_mul_ps(e, vrecip));
                    }
                }
                /* Tail: skaler */
                for (; kk < k_end; ++kk) {
                    float mx = -1e30f;
                    for (int64_t a = 0; a < axis_dim; ++a) {
                        float v = ip[o * axis_dim * inner + a * inner + kk];
                        if (v > mx) mx = v;
                    }
                    float sum = 0.0f;
                    for (int64_t a = 0; a < axis_dim; ++a) {
                        float e = std::exp(ip[o * axis_dim * inner + a * inner + kk] - mx);
                        op_[o * axis_dim * inner + a * inner + kk] = e;
                        sum += e;
                    }
                    float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
                    for (int64_t a = 0; a < axis_dim; ++a) {
                        op_[o * axis_dim * inner + a * inner + kk] *= inv;
                    }
                }
            }
        });

        const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
        if (out_idx >= 0) intermediate_[static_cast<size_t>(out_idx)] = std::move(out);
        return true;
    }

    /* ---------------- TopK ---------------- */
    case OpType::TopK: {
        if (n_ins < 2) return false;
        const Tensor& input = *ins_ptr[0];
        const Tensor& k_t   = *ins_ptr[1];

        int64_t k = 0;
        if (k_t.dtype() == DType::I64)      k = k_t.data<int64_t>()[0];
        else if (k_t.dtype() == DType::I32) k = k_t.data<int32_t>()[0];
        else                                k = static_cast<int64_t>(k_t.data<float>()[0]);

        const int32_t rank = input.rank();
        int32_t ax = node.params.axis;
        if (ax < 0) ax += rank;
        if (ax < 0 || ax >= rank) return false;

        const int64_t axis_dim = input.dim(ax);
        if (k <= 0 || k > axis_dim) k = axis_dim;

        /* Output shape: input shape with axis_dim = k */
        int64_t out_dims[kMaxNdim];
        for (int32_t d = 0; d < rank; ++d) out_dims[d] = input.dim(d);
        out_dims[ax] = k;
        Shape os(out_dims, rank);

        Tensor values  = acquire_output_buffer(os, DType::F32);
        Tensor indices = acquire_output_buffer(os, DType::I32);
        if (values.is_empty() || indices.is_empty()) return false;

        /* TopK hesapla */
        int64_t outer = 1;
        for (int32_t d = 0; d < ax; ++d) outer *= input.dim(d);
        int64_t inner = 1;
        for (int32_t d = ax + 1; d < rank; ++d) inner *= input.dim(d);

        const float*   in = input.data<float>();
        float*         vo = values.data<float>();
        int32_t*       io = indices.data<int32_t>();

        struct KV { float v; int32_t i; };
        std::vector<KV> buf(axis_dim);

        const bool largest = (node.params.fused_activation != 0)
                             ? true : true;   /* ONNX default largest=1 */

        for (int64_t o = 0; o < outer; ++o) {
            for (int64_t in_i = 0; in_i < inner; ++in_i) {
                const float* base = in + (o * axis_dim) * inner + in_i;
                for (int64_t a = 0; a < axis_dim; ++a) {
                    buf[a].v = base[a * inner];
                    buf[a].i = static_cast<int32_t>(a);
                }
                if (largest) {
                    std::partial_sort(buf.begin(), buf.begin() + k, buf.end(),
                        [](const KV& x, const KV& y) { return x.v > y.v; });
                } else {
                    std::partial_sort(buf.begin(), buf.begin() + k, buf.end(),
                        [](const KV& x, const KV& y) { return x.v < y.v; });
                }
                float*   vb = vo + (o * k) * inner + in_i;
                int32_t* ib = io + (o * k) * inner + in_i;
                for (int64_t j = 0; j < k; ++j) {
                    vb[j * inner] = buf[j].v;
                    ib[j * inner] = buf[j].i;
                }
            }
        }

        if (node.outputs.size() >= 1 && node.outputs[0] >= 0)
            intermediate_[static_cast<size_t>(node.outputs[0])] = std::move(values);
        if (node.outputs.size() >= 2 && node.outputs[1] >= 0)
            intermediate_[static_cast<size_t>(node.outputs[1])] = std::move(indices);
        return true;
    }

    /* ---------------- GatherElements ---------------- */
    case OpType::GatherElements: {
        if (n_ins < 2) return false;
        const Tensor& data    = *ins_ptr[0];
        const Tensor& indices = *ins_ptr[1];

        const int32_t rank = data.rank();
        int32_t ax = node.params.axis;
        if (ax < 0) ax += rank;
        if (ax < 0 || ax >= rank) return false;

        Tensor out = acquire_output_buffer(indices.shape(), DType::F32);
        if (out.is_empty()) return false;

        const int64_t axis_dim = data.dim(ax);
        int64_t outer = 1;
        for (int32_t d = 0; d < ax; ++d) outer *= data.dim(d);
        int64_t inner = 1;
        for (int32_t d = ax + 1; d < rank; ++d) inner *= data.dim(d);
        const int64_t idx_axis_dim = indices.dim(ax);

        const float*   src = data.data<float>();
        float*         dst = out.data<float>();

        for (int64_t o = 0; o < outer; ++o) {
            for (int64_t a = 0; a < idx_axis_dim; ++a) {
                for (int64_t i = 0; i < inner; ++i) {
                    int32_t gi = get_tensor_int(&indices, (o * idx_axis_dim + a) * inner + i);
                    if (gi < 0) gi += static_cast<int32_t>(axis_dim);
                    if (gi < 0 || gi >= axis_dim) gi = 0;
                    dst[(o * idx_axis_dim + a) * inner + i] =
                        src[(o * axis_dim + gi) * inner + i];
                }
            }
        }

        const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
        if (out_idx >= 0) intermediate_[static_cast<size_t>(out_idx)] = std::move(out);
        return true;
    }