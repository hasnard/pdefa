/* ============================================================
 *  NHWC AXIS REMAP HELPER
 *  NCHW → NHWC permütasyonu: [0, 3, 1, 2]
 * ============================================================ */





    /* ---------------- Stack (numpy.stack) ---------------- */
    case OpType::Stack: {
        if (n_ins < 1) return false;
        const Shape& s0 = ins_ptr[0]->shape();
        const int32_t rank = s0.rank();
        if (rank < 0 || rank >= kMaxNdim) return false;

        for (size_t i = 1; i < n_ins; ++i) {
            if (ins_ptr[i]->shape() != s0) return false;
        }

        const int32_t new_rank = rank + 1;
        int32_t ax = node.params.axis;
        if (ax < 0) ax += new_rank;
        if (ax < 0 || ax >= new_rank) return false;

        int64_t out_dims[kMaxNdim];
        int32_t j = 0;
        for (int32_t d = 0; d < new_rank; ++d) {
            if (d == ax) out_dims[d] = static_cast<int64_t>(n_ins);
            else         out_dims[d] = s0.dim(j++);
        }
        Shape os(out_dims, new_rank);

        Tensor out = acquire_output_buffer(os, DType::F32);
        if (out.is_empty()) return false;

        int64_t outer = 1;
        for (int32_t d = 0; d < ax; ++d) outer *= out_dims[d];
        const int64_t inner = (outer > 0) ? (s0.numel() / outer) : s0.numel();

        float* dst_base = out.data<float>();
        for (int64_t o = 0; o < outer; ++o) {
            for (size_t n = 0; n < n_ins; ++n) {
                const float* src = ins_ptr[n]->data<float>() + o * inner;
                float*       dst = dst_base + (o * (int64_t)n_ins + (int64_t)n) * inner;
                std::memcpy(dst, src, static_cast<size_t>(inner) * sizeof(float));
            }
        }

        const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
        if (out_idx >= 0) intermediate_[static_cast<size_t>(out_idx)] = std::move(out);
        return true;
    }











    /* ---------------- Concat ---------------- */
    case OpType::Concat: {
        if (n_ins == 0) return false;
        const Shape& s0 = ins_ptr[0]->shape();
        const int32_t rank = s0.rank();
        int32_t ax = (node.params.axis < 0) ? node.params.axis + rank : node.params.axis;

            /* NHWC remap — SADECE C split icin (YOLOv10 attention axes=2 NCHW kalir) */
            if (rank == 4 && ax == 1) {
                ax = 3;
            }

        if (ax < 0 || ax >= rank) return false;

        int64_t dims[kMaxNdim];
        for (int32_t i = 0; i < rank; ++i) dims[i] = s0.dim(i);
        dims[ax] = 0;
        for (size_t i = 0; i < n_ins; ++i) dims[ax] += ins_ptr[i]->dim(ax);
        const Shape cs(dims, rank);

        Tensor out = acquire_output_buffer(cs, DType::F32);
        if (out.is_empty()) return false;

        ops::concat(ins_ptr, static_cast<int32_t>(n_ins), out, ax);

        const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
        if (out_idx >= 0) intermediate_[static_cast<size_t>(out_idx)] = std::move(out);
        return true;
    }

    /* ---------------- Upsample / Resize (NHWC) ---------------- */
    case OpType::Upsample:
    case OpType::Resize: {
        if (n_ins == 0) return false;
        const Tensor& in = *ins_ptr[0];
        const Shape in_shape = in.shape();
        if (in_shape.rank() != 4) return false;

        const int64_t N_in = in_shape.dim(0);
        const int64_t H_in = in_shape.dim(1);
        const int64_t W_in = in_shape.dim(2);
        const int64_t C_in = in_shape.dim(3);

        int64_t H = 0, W = 0;

        if (node.op == OpType::Upsample) {
            H = H_in * node.params.scale_h;
            W = W_in * node.params.scale_w;
        } else {
            H = node.params.target_h;
            W = node.params.target_w;

            if ((H <= 0 || W <= 0) &&
                node.params.scale_h > 1 && node.params.scale_w > 1) {
                H = H_in * node.params.scale_h;
                W = W_in * node.params.scale_w;
            }

            if (H <= 0 || W <= 0) {
                for (size_t k = 1; k < n_ins && k < 4; ++k) {
                    const Tensor* t = ins_ptr[k];
                    if (!t || t->is_empty()) continue;

                    if (t->numel() == 4) {
                        const float s2 = get_tensor_float(t, 2);
                        const float s3 = get_tensor_float(t, 3);

                        if (s2 > 8.0f && s3 > 8.0f) {
                            H = static_cast<int64_t>(s2 + 0.5f);
                            W = static_cast<int64_t>(s3 + 0.5f);
                        } else {
                            H = static_cast<int64_t>(
                                static_cast<float>(H_in) * s2 + 0.5f);
                            W = static_cast<int64_t>(
                                static_cast<float>(W_in) * s3 + 0.5f);
                        }
                        break;
                    }
                    else if (t->numel() == 2) {
                        const float s0 = get_tensor_float(t, 0);
                        const float s1 = get_tensor_float(t, 1);
                        if (s0 > 8.0f && s1 > 8.0f) {
                            H = static_cast<int64_t>(s0 + 0.5f);
                            W = static_cast<int64_t>(s1 + 0.5f);
                        } else {
                            H = static_cast<int64_t>(
                                static_cast<float>(H_in) * s0 + 0.5f);
                            W = static_cast<int64_t>(
                                static_cast<float>(W_in) * s1 + 0.5f);
                        }
                        break;
                    }
                }
            }

            if (H <= 0 || W <= 0) {
                H = H_in * 2;
                W = W_in * 2;
            }
        }
        if (H <= 0 || W <= 0) return false;

        int64_t dims[4] = { N_in, H, W, C_in };
        Shape os(dims, 4);

        Tensor out = acquire_output_buffer(os, DType::F32);
        if (out.is_empty()) return false;

        const float* in_data = in.data<float>();
        float* out_data = out.data<float>();

        const float scale_y = static_cast<float>(H_in) / static_cast<float>(H);
        const float scale_x = static_cast<float>(W_in) / static_cast<float>(W);
        const size_t c_bytes = static_cast<size_t>(C_in) * sizeof(float);

        auto& pool = runtime::GlobalPool::get();
        pool.parallel_for(0, N_in * H * W, [&](int64_t start, int64_t end) {
            for (int64_t idx = start; idx < end; ++idx) {
                const int64_t w_out = idx % W;
                const int64_t t1 = idx / W;
                const int64_t h_out = t1 % H;
                const int64_t n = t1 / H;

                const int64_t h_in = std::min<int64_t>(
                    static_cast<int64_t>(h_out * scale_y), H_in - 1);
                const int64_t w_in = std::min<int64_t>(
                    static_cast<int64_t>(w_out * scale_x), W_in - 1);

                const float* src = in_data + ((n * H_in + h_in) * W_in + w_in) * C_in;
                float*       dst = out_data + ((n * H + h_out) * W + w_out) * C_in;
                std::memcpy(dst, src, c_bytes);
            }
        });

        const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
        if (out_idx >= 0) intermediate_[static_cast<size_t>(out_idx)] = std::move(out);
        return true;
    }

    /* ---------------- Mod ---------------- */
    case OpType::Mod: {
        if (n_ins < 2) return false;
        const Tensor& a = *ins_ptr[0];
        const Tensor& b = *ins_ptr[1];

        const int64_t na = a.numel();
        const int64_t nb = b.numel();
        if (na != nb && nb != 1) return false;

        Tensor out = acquire_output_buffer(a.shape(), DType::F32);
        if (out.is_empty()) return false;

        const float* ap = a.data<float>();
        const float* bp = b.data<float>();
        float*       op_ = out.data<float>();

        if (nb == 1) {
            const float bv = bp[0];
            for (int64_t i = 0; i < na; ++i) {
                float r = ap[i] - bv * std::floor(ap[i] / bv);
                op_[i] = r;
            }
        } else {
            for (int64_t i = 0; i < na; ++i) {
                float r = ap[i] - bp[i] * std::floor(ap[i] / bp[i]);
                op_[i] = r;
            }
        }

        const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
        if (out_idx >= 0) intermediate_[static_cast<size_t>(out_idx)] = std::move(out);
        return true;
    }

    /* ---------------- ReduceMax ---------------- */
    case OpType::ReduceMax: {
        if (n_ins < 1) return false;
        const Tensor& in = *ins_ptr[0];

        const int32_t rank = in.rank();
        int32_t ax = node.params.axis;
        if (ax < 0) ax += rank;
        if (ax < 0 || ax >= rank) return false;

        int64_t outer = 1;
        for (int32_t d = 0; d < ax; ++d) outer *= in.dim(d);
        const int64_t axis_dim = in.dim(ax);
        int64_t inner = 1;
        for (int32_t d = ax + 1; d < rank; ++d) inner *= in.dim(d);

        /* keepdims dosyadan gelmiyor: OpParams raw-struct olarak okunuyor
         * (custom_loader.cpp: read_bytes(&params, sizeof(OpParams))),
         * mevcut .engine formatında keepdims alanı yok — eklemek formatı
         * bozar. Motor rank-4 NHWC varsayımı üzerine kurulu; YOLO'nun
         * "Unsqueeze×N → Concat → ReduceSum/Max(axis=0)" pattern'i ONNX'te
         * keepdims=0'dır. Sezgisel:
         *   rank>=5 & ax==0 → false (indirgenen ekseni tamamen düşür)
         *   aksi halde     → true  (ONNX default, rank korunur) */
        const bool keep = !(rank >= 5 && ax == 0);

        int64_t out_dims[kMaxNdim];
        int32_t orank = 0;
        if (keep) {
            for (int32_t d = 0; d < rank; ++d)
                out_dims[orank++] = (d == ax) ? 1 : in.dim(d);
        } else {
            for (int32_t d = 0; d < rank; ++d) {
                if (d == ax) continue;
                out_dims[orank++] = in.dim(d);
            }
            if (orank == 0) { out_dims[0] = 1; orank = 1; }
        }
        Shape os(out_dims, orank);

        Tensor out = acquire_output_buffer(os, DType::F32);
        if (out.is_empty()) return false;

        const float* ip = in.data<float>();
        float*       op_ = out.data<float>();

        for (int64_t o = 0; o < outer; ++o) {
            for (int64_t i = 0; i < inner; ++i) {
                float mx = -3.402823466e+38f;
                const float* base = ip + (o * axis_dim) * inner + i;
                for (int64_t a = 0; a < axis_dim; ++a) {
                    const float v = base[a * inner];
                    if (v > mx) mx = v;
                }
                op_[o * inner + i] = mx;
            }
        }

        const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
        if (out_idx >= 0) intermediate_[static_cast<size_t>(out_idx)] = std::move(out);
        return true;
    }

    /* ---------------- ArgMax ---------------- */
    case OpType::ArgMax: {
        if (n_ins < 1) return false;
        const Tensor& in = *ins_ptr[0];

        const int32_t rank = in.rank();
        int32_t ax = node.params.axis;
        if (ax < 0) ax += rank;
        if (ax < 0 || ax >= rank) return false;

        int64_t outer = 1;
        for (int32_t d = 0; d < ax; ++d) outer *= in.dim(d);
        const int64_t axis_dim = in.dim(ax);
        int64_t inner = 1;
        for (int32_t d = ax + 1; d < rank; ++d) inner *= in.dim(d);

        /* ONNX default keepdims=1; rank>=5 & ax==0 sezgiseli Reduce ailesiyle aynı */
        const bool keep = !(rank >= 5 && ax == 0);

        int64_t out_dims[kMaxNdim];
        int32_t orank = 0;
        if (keep) {
            for (int32_t d = 0; d < rank; ++d)
                out_dims[orank++] = (d == ax) ? 1 : in.dim(d);
        } else {
            for (int32_t d = 0; d < rank; ++d) {
                if (d == ax) continue;
                out_dims[orank++] = in.dim(d);
            }
            if (orank == 0) { out_dims[0] = 1; orank = 1; }
        }
        Shape os(out_dims, orank);

        Tensor out = acquire_output_buffer(os, DType::I32);
        if (out.is_empty()) return false;

        const float* ip = in.data<float>();
        int32_t*     op_ = out.data<int32_t>();

        for (int64_t o = 0; o < outer; ++o) {
            for (int64_t i = 0; i < inner; ++i) {
                const float* base = ip + (o * axis_dim) * inner + i;
                float best = base[0];
                int32_t best_idx = 0;
                for (int64_t a = 1; a < axis_dim; ++a) {
                    const float v = base[a * inner];
                    if (v > best) { best = v; best_idx = static_cast<int32_t>(a); }
                }
                op_[o * inner + i] = best_idx;
            }
        }

        const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
        if (out_idx >= 0) intermediate_[static_cast<size_t>(out_idx)] = std::move(out);
        return true;
    }

    /* ---------------- ReduceSum ---------------- */
case OpType::ReduceSum: {
    if (n_ins < 1) return false;
    const Tensor& in = *ins_ptr[0];

    const int32_t rank = in.rank();

    /* ONNX-13+: axes ikinci input olarak gelir; params.axis çoğu zaman
     * default 0'dır. İkinci input varsa onu tercih et. */
    int32_t ax = node.params.axis;
    if (n_ins >= 2 && ins_ptr[1] && !ins_ptr[1]->is_empty() &&
        ins_ptr[1]->numel() > 0) {
        ax = static_cast<int32_t>(get_tensor_int(ins_ptr[1], 0));
    }
    if (ax < 0) ax += rank;
    if (ax < 0 || ax >= rank) return false;

    int64_t outer = 1;
    for (int32_t d = 0; d < ax; ++d) outer *= in.dim(d);
    const int64_t axis_dim = in.dim(ax);
    int64_t inner = 1;
    for (int32_t d = ax + 1; d < rank; ++d) inner *= in.dim(d);

    /* keepdims dosyadan gelmiyor: OpParams raw-struct olarak okunuyor
     * (custom_loader.cpp: read_bytes(&params, sizeof(OpParams))),
     * mevcut .engine formatında keepdims alanı yok — eklemek formatı
     * bozar. Motor rank-4 NHWC varsayımı üzerine kurulu; YOLO'nun
     * "Unsqueeze×N → Concat → ReduceSum(axis=0)" stack+sum pattern'i
     * ONNX'te keepdims=0'dır. Sezgisel:
     *   rank>=5 & ax==0 → false (indirgenen ekseni tamamen düşür)
     *   aksi halde     → true  (ONNX default, rank korunur) */
    const bool keep = !(rank >= 5 && ax == 0);

    int64_t out_dims[kMaxNdim];
    int32_t orank = 0;
    if (keep) {
        for (int32_t d = 0; d < rank; ++d)
            out_dims[orank++] = (d == ax) ? 1 : in.dim(d);
    } else {
        for (int32_t d = 0; d < rank; ++d) {
            if (d == ax) continue;
            out_dims[orank++] = in.dim(d);
        }
        if (orank == 0) { out_dims[0] = 1; orank = 1; }
    }
    Shape os(out_dims, orank);

    Tensor out = acquire_output_buffer(os, DType::F32);
    if (out.is_empty()) return false;

    const float* ip = in.data<float>();
    float*       op_ = out.data<float>();

    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t i = 0; i < inner; ++i) {
            float s = 0.0f;
            const float* base = ip + (o * axis_dim) * inner + i;
            for (int64_t a = 0; a < axis_dim; ++a) {
                s += base[a * inner];
            }
            op_[o * inner + i] = s;
        }
    }

    const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
    if (out_idx >= 0) intermediate_[static_cast<size_t>(out_idx)] = std::move(out);
    return true;
}









    /* ---------------- Tile ---------------- */
    case OpType::Tile: {
        if (n_ins < 2) return false;
        const Tensor& in     = *ins_ptr[0];
        const Tensor& reps_t = *ins_ptr[1];
        const int32_t in_rank = in.rank();
        if (in_rank < 1 || in_rank > 8) return false;

        const int64_t reps_n = reps_t.numel();
        if (reps_n < 1 || reps_n > 8) return false;

        /* Numpy tile semantiği:
         *  - Çıktı rank = max(in_rank, reps_n)
         *  - Kısa olan taraf başa 1'lerle hizalanır.
         */
        const int32_t out_rank = (in_rank > (int32_t)reps_n) ? in_rank : (int32_t)reps_n;

        /* Etkin input boyutları (başa 1 ekleyerek out_rank'e uzat) */
        int64_t in_dims[8];
        const int32_t in_pad = out_rank - in_rank;
        for (int32_t d = 0; d < out_rank; ++d) {
            in_dims[d] = (d < in_pad) ? 1 : in.dim(d - in_pad);
        }

        /* Etkin reps (başa 1 ekleyerek out_rank'e uzat) */
        int64_t reps_v[8];
        const int64_t reps_pad = static_cast<int64_t>(out_rank) - reps_n;
        for (int32_t d = 0; d < out_rank; ++d) {
            if (d < reps_pad) {
                reps_v[d] = 1;
            } else {
                int64_t r = get_tensor_int(&reps_t, d - reps_pad);
                if (r < 1) r = 1;
                reps_v[d] = r;
            }
        }

        /* Çıktı boyutları */
        int64_t out_dims[8];
        for (int32_t d = 0; d < out_rank; ++d) {
            out_dims[d] = in_dims[d] * reps_v[d];
        }
        Shape os(out_dims, out_rank);
        Tensor out = acquire_output_buffer(os, in.dtype());
        if (out.is_empty()) return false;

        const float* ip = in.data<float>();
        float*       op = out.data<float>();

        /* Input stride'ları — in_dims (padded) üzerinden */
        int64_t in_strides[8];
        in_strides[out_rank - 1] = 1;
        for (int32_t d = out_rank - 2; d >= 0; --d)
            in_strides[d] = in_strides[d + 1] * in_dims[d + 1];

        const int64_t n_out = out.numel();
        int64_t o_idx[8];
        for (int64_t i = 0; i < n_out; ++i) {
            int64_t t   = i;
            int64_t src = 0;
            for (int32_t d = out_rank - 1; d >= 0; --d) {
                o_idx[d] = t % out_dims[d];
                t /= out_dims[d];
                const int64_t id = in_dims[d];
                src += (id > 0 ? (o_idx[d] % id) : 0) * in_strides[d];
            }
            op[i] = ip[src];
        }

        const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
        if (out_idx >= 0) intermediate_[static_cast<size_t>(out_idx)] = std::move(out);
        return true;
    }


    /* ---------------- Reshape ---------------- */
    case OpType::Reshape: {
        if (n_ins == 0) return false;
        const Tensor& in = *ins_ptr[0];
        if (in.is_empty()) return false;

        /* ============================================================
         *  CHANNEL SHUFFLE — 4D↔5D (NHWC motor düzeni)
         *  ONNX hedefi [N, g, C/g, H, W] → motor hedefi [N, H, W, g, C/g]
         *  Aynı flat memory, sadece shape etiketi değişir.
         * ============================================================ */
        if (in.rank() == 4 && n_ins >= 2 && ins_ptr[1] &&
            !ins_ptr[1]->is_empty() && ins_ptr[1]->numel() == 5) {
            int64_t tgt[5];
            for (int i = 0; i < 5; ++i) tgt[i] = get_tensor_int(ins_ptr[1], i);
            const int64_t N_in = in.dim(0), H_in = in.dim(1),
                          W_in = in.dim(2), C_in = in.dim(3);
            const int64_t N_o = tgt[0], g = tgt[1], Cp = tgt[2],
                          H_o = tgt[3], W_o = tgt[4];
            if (N_in == N_o && H_in == H_o && W_in == W_o && g * Cp == C_in) {
                int64_t motor_dims[5] = { N_o, H_o, W_o, g, Cp };
                Shape motor_os(motor_dims, 5);
                Tensor out = acquire_output_buffer(motor_os, DType::F32);
                if (out.is_empty()) return false;
                std::memcpy(out.data(), in.data(), in.nbytes());
                const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
                if (out_idx >= 0)
                    intermediate_[static_cast<size_t>(out_idx)] = std::move(out);
                return true;
            }
        }
        if (in.rank() == 5 && n_ins >= 2 && ins_ptr[1] &&
            !ins_ptr[1]->is_empty() && ins_ptr[1]->numel() == 4) {
            int64_t tgt[4];
            for (int i = 0; i < 4; ++i) tgt[i] = get_tensor_int(ins_ptr[1], i);
            const int64_t N_in = in.dim(0), H_in = in.dim(1),
                          W_in = in.dim(2), g = in.dim(3), Cp = in.dim(4);
            const int64_t N_o = tgt[0], C_o = tgt[1], H_o = tgt[2], W_o = tgt[3];
            if (N_in == N_o && H_in == H_o && W_in == W_o && g * Cp == C_o) {
                int64_t motor_dims[4] = { N_o, H_o, W_o, C_o };
                Shape motor_os(motor_dims, 4);
                Tensor out = acquire_output_buffer(motor_os, DType::F32);
                if (out.is_empty()) return false;
                std::memcpy(out.data(), in.data(), in.nbytes());
                const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
                if (out_idx >= 0)
                    intermediate_[static_cast<size_t>(out_idx)] = std::move(out);
                return true;
            }
        }

        int64_t new_dims[8];
        int32_t new_rank = 0;

        if (n_ins >= 2 && !ins_ptr[1]->is_empty()) {
            const Tensor& shape_t = *ins_ptr[1];
            const int64_t n = shape_t.numel();
            if (n < 1 || n > 8) return false;
            for (int64_t d = 0; d < n; ++d) {
                int64_t s = get_tensor_int(&shape_t, d);
                if (s == 0) s = in.dim(d);
                new_dims[d] = s;
            }
            new_rank = (int32_t)n;

            int64_t known = 1;
            int neg_idx = -1;
            for (int i = 0; i < new_rank; ++i) {
                if (new_dims[i] == -1) neg_idx = i;
                else known *= new_dims[i];
            }
            if (neg_idx >= 0) {
                const int64_t total = in.numel();
                new_dims[neg_idx] = (known > 0) ? (total / known) : 0;
            }
        } else {
            const int64_t batch = in.dim(0);
            const int64_t flat  = (batch > 0) ? (in.numel() / batch) : in.numel();
            new_dims[0] = batch;
            new_dims[1] = flat;
            new_rank = 2;
        }

        /* ============================================================
         *  4D → 4D Reshape özel durumu (YOLOv10 attention):
         *  ONNX hedefi [N', C', H', W'] (NCHW). Motor NHWC tutar.
         * ============================================================ */
        if (in.rank() == 4 && new_rank == 4 && in.dtype() == DType::F32) {
            const int64_t N_in = in.dim(0), H_in = in.dim(1),
                          W_in = in.dim(2), C_in = in.dim(3);
            const int64_t N_out = new_dims[0], C_out = new_dims[1],
                          H_out = new_dims[2], W_out = new_dims[3];

            if (N_out > 0 && C_out > 0 && H_out > 0 && W_out > 0 &&
                N_in * H_in * W_in * C_in == N_out * C_out * H_out * W_out) {

                int64_t nhwc_dims[4] = { N_out, H_out, W_out, C_out };
                Shape nhwc_os(nhwc_dims, 4);

                Tensor nhwc_out = acquire_output_buffer(nhwc_os, DType::F32);
                if (nhwc_out.is_empty()) return false;

                const float* ip = in.data<float>();
                float*       op = nhwc_out.data<float>();

                for (int64_t n = 0; n < N_out; ++n) {
                    for (int64_t c = 0; c < C_out; ++c) {
                        for (int64_t h = 0; h < H_out; ++h) {
                            for (int64_t w = 0; w < W_out; ++w) {
                                const int64_t nchw = c * (H_out * W_out) + h * W_out + w;
                                int64_t t = nchw;
                                const int64_t w_i = t % W_in; t /= W_in;
                                const int64_t h_i = t % H_in; t /= H_in;
                                const int64_t c_i = t % C_in; t /= C_in;
                                const int64_t n_i = t;

                                const int64_t in_idx =
                                    ((n_i * H_in + h_i) * W_in + w_i) * C_in + c_i;
                                const int64_t out_off =
                                    ((n * H_out + h) * W_out + w) * C_out + c;
                                op[out_off] = ip[in_idx];
                            }
                        }
                    }
                }

                const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
                if (out_idx >= 0)
                    intermediate_[static_cast<size_t>(out_idx)] = std::move(nhwc_out);
                return true;
            }
        }

        Shape os(new_dims, new_rank);
        Tensor out = acquire_output_buffer(os, in.dtype());
        if (out.is_empty()) return false;

        /* ============================================================
         *  4D → 3D özel durum: motor NHWC [N,H,W,C] → ONNX [N,C,H*W] veya [N,C,S]
         *  Örnek (DFL Reshape_1): motor [1,8400,4,1] → ONNX [1,4,8400]
         *  Flat memcpy yanlış; anchor-major'dan side-major'a permute şart.
         * ============================================================ */
        if (in.rank() == 4 && new_rank == 3 && in.dtype() == DType::F32) {
            const int64_t N_in = in.dim(0);
            const int64_t H_in = in.dim(1);
            const int64_t W_in = in.dim(2);
            const int64_t C_in = in.dim(3);

            const int64_t N_out = new_dims[0];
            const int64_t C_out = new_dims[1];
            const int64_t S_out = new_dims[2];

            /* Sadece C_in == 1 && W_in == C_out && H_in * W_in == C_out * S_out
             * gibi DFL-benzeri pattern'lerde uygula (güvenli tespit). */
            if (C_in == 1 &&
                N_in == N_out &&
                W_in == C_out &&
                H_in == S_out) {

                Shape os(new_dims, 3);
                Tensor out = acquire_output_buffer(os, DType::F32);
                if (out.is_empty()) return false;

                const float* ip = in.data<float>();
                float*       op = out.data<float>();

                /* Motor [N,H,W,1] → ONNX [N,W,H] flat:
                 *   onnx_flat[c_out * S_out + s_out] = motor_flat[h * W + w]
                 *   c_out = w, s_out = h  (DFL semantiği)
                 */
                for (int64_t h = 0; h < H_in; ++h) {
                    for (int64_t w = 0; w < W_in; ++w) {
                        const int64_t in_off  = (h * W_in + w);         // C=1, N=1
                        const int64_t out_off = w * S_out + h;          // [N=1, C, S]
                        op[out_off] = ip[in_off];
                    }
                }

                const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
                if (out_idx >= 0) intermediate_[static_cast<size_t>(out_idx)] = std::move(out);
                return true;
            }
        }

        /* NHWC → NCHW-reshape tespiti (diğer 4D → non-4D durumları). */
        bool need_convert = false;
        if (in.rank() == 4 && new_rank != 4 && in.dtype() == DType::F32) {
            const int64_t c = in.dim(3);
            const int64_t h = in.dim(1);
            const int64_t w = in.dim(2);

            if (new_rank == 2 && new_dims[1] == (c * h * w)) {
                need_convert = true;
            } else if (new_rank >= 2 && new_dims[1] == c) {
                int64_t target_spatial = 1;
                for (int32_t d = 2; d < new_rank; ++d) target_spatial *= new_dims[d];
                if (h * w == target_spatial) {
                    need_convert = true;
                }
            }
        }
        if (need_convert) {
            Tensor nchw = optimizer::nhwc_to_nchw(in);
            if (!nchw.is_empty() && nchw.numel() == out.numel()) {
                std::memcpy(out.data(), nchw.data(), nchw.nbytes());
            } else {
                std::memcpy(out.data(), in.data(), in.nbytes());
            }
        } else {
            std::memcpy(out.data(), in.data(), in.nbytes());
        }

        const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
        if (out_idx >= 0) intermediate_[static_cast<size_t>(out_idx)] = std::move(out);
        return true;
    }

    /* ---------------- Shape ---------------- */
    case OpType::Shape: {
        if (n_ins == 0) return false;
        const Tensor& in = *ins_ptr[0];
        const int64_t n = in.rank();
        int64_t dims[1] = { n };
        Shape os(dims, 1);
        Tensor out = acquire_output_buffer(os, DType::F32);
        if (out.is_empty()) return false;
        float* p = out.data<float>();

        if (n == 4) {
            p[0] = (float)in.dim(0);   /* N */
            p[1] = (float)in.dim(3);   /* C */
            p[2] = (float)in.dim(1);   /* H */
            p[3] = (float)in.dim(2);   /* W */
        } else {
            for (int64_t i = 0; i < n; ++i) p[i] = (float)in.dim(i);
        }

        const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
        if (out_idx >= 0) intermediate_[out_idx] = std::move(out);
        return true;
    }

    /* ---------------- Slice (NHWC axis remap — partial) ---------------- */
    case OpType::Slice: {
        if (n_ins < 3) return false;
        const Tensor& data     = *ins_ptr[0];
        const Tensor& starts_t = *ins_ptr[1];
        const Tensor& ends_t   = *ins_ptr[2];
        const Tensor* axes_t   = (n_ins > 3 && ins_ptr[3] &&
                                  !ins_ptr[3]->is_empty() &&
                                  ins_ptr[3]->numel() > 0) ? ins_ptr[3] : nullptr;

        const Tensor* steps_t  = (n_ins > 4 && ins_ptr[4] && !ins_ptr[4]->is_empty() && ins_ptr[4]->numel() > 0) ? ins_ptr[4] : nullptr;

        const int64_t n_slice = starts_t.numel();
        const int64_t rank    = data.rank();
        if (n_slice < 1 || rank < 1 || rank > 8) return false;

        int64_t out_dims[8], start_v[8], step_v[8];
        for (int64_t d = 0; d < rank; ++d) {
            out_dims[d] = data.dim(d);
            start_v[d]  = 0;
            step_v[d]   = 1;
        }

        for (int64_t k = 0; k < n_slice; ++k) {
            int64_t ax = axes_t ? get_tensor_int(axes_t, k) : k;
            if (ax < 0) ax += rank;

            const int64_t ax_orig = ax;

        /* Full NHWC remap (ONNX NCHW ↔ motor NHWC)
         *   ONNX axis=1 (C) → motor axis=3
         *   ONNX axis=2 (H) → motor axis=1
         *   ONNX axis=3 (W) → motor axis=2
         * Sonradan gelen "adaptive revert" bu remap'i, dilim motor
         * eksenine sığmıyorsa geri alır. */
        if (rank == 4) {
            if (ax == 1)      ax = 3;
            else if (ax == 2) ax = 1;
            else if (ax == 3) ax = 2;
        }

            int64_t s  = get_tensor_int(&starts_t, k);
            int64_t e  = get_tensor_int(&ends_t, k);

            /* ADAPTIVE: remap sonrası eksende sınır aşımı varsa ama orijinal
             * eksende yoksa, remap yanlış demektir → orijinal eksene dön.
             * v12 attention: data=[4,2,96,400] axes=2 ends=32
             *   remap(2→1): dim=2, ends=32 > 2  → yanlış
             *   orijinal:   dim=96, ends=32 ≤ 96 → doğru
             */
        if (ax != ax_orig && rank == 4) {
              const int64_t dim_remap = data.dim(ax);
              const int64_t dim_orig  = data.dim(ax_orig);
              if (e > dim_remap * 2 && e <= dim_orig) {
                  ax = ax_orig;
              }
            }

            if (ax < 0 || ax >= rank) continue;

            int64_t st = steps_t ? get_tensor_int(steps_t, k) : 1;
            if (st == 0) st = 1;

            const int64_t dim_len = data.dim(ax);
            if (s < 0) s += dim_len;
            if (e < 0) e += dim_len;
            if (s < 0) s = 0;
            if (s > dim_len) s = dim_len;
            if (e < 0) e = 0;
            if (e > dim_len) e = dim_len;
            if (e < s) e = s;

            start_v[ax] = s;
            step_v[ax]  = st;
            out_dims[ax] = (st > 0) ? ((e - s + st - 1) / st) : 0;
        }

        Shape os(out_dims, (int32_t)rank);
        Tensor out = acquire_output_buffer(os, data.dtype());
        if (out.is_empty()) return false;

        const float* dp = data.data<float>();
        float* op_ = out.data<float>();

        int  slice_axis  = -1;
        bool fast_path   = true;

        for (int64_t d = 0; d < rank; ++d) {
            if (out_dims[d] == data.dim(d) && start_v[d] == 0 && step_v[d] == 1) continue;
            if (slice_axis != -1) { fast_path = false; break; }
            if (step_v[d] != 1)   { fast_path = false; break; }
            slice_axis = (int)d;
        }

        if (fast_path && slice_axis >= 0) {
            int64_t outer = 1;
            for (int64_t d = 0; d < slice_axis; ++d) outer *= data.dim(d);
            int64_t inner = 1;
            for (int64_t d = slice_axis + 1; d < rank; ++d) inner *= data.dim(d);

            const int64_t slice_start     = start_v[slice_axis];
            const int64_t slice_len       = out_dims[slice_axis];
            const int64_t in_outer_stride = data.dim(slice_axis) * inner;
            const int64_t copy_bytes      = slice_len * inner * sizeof(float);

            const char* src_base = reinterpret_cast<const char*>(dp + slice_start * inner);
            char* dst_base = reinterpret_cast<char*>(op_);

            for (int64_t o = 0; o < outer; ++o) {
                std::memcpy(dst_base + o * copy_bytes,
                            src_base + o * in_outer_stride * sizeof(float),
                            copy_bytes);
            }
        } else if (fast_path && slice_axis == -1) {
            std::memcpy(op_, dp, data.nbytes());
        } else {
            const int64_t n_out = out.numel();
            int64_t o_idx[8], s_idx[8];
            for (int64_t i = 0; i < n_out; ++i) {
                int64_t t = i;
                for (int64_t d = rank - 1; d >= 0; --d) {
                    o_idx[d] = t % out_dims[d];
                    t /= out_dims[d];
                }
                for (int64_t d = 0; d < rank; ++d)
                    s_idx[d] = start_v[d] + o_idx[d] * step_v[d];
                int64_t src = 0;
                for (int64_t d = 0; d < rank; ++d)
                    src = src * data.dim(d) + s_idx[d];
                op_[i] = dp[src];
            }
        }

        const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
        if (out_idx >= 0) intermediate_[out_idx] = std::move(out);
        return true;
    }

    /* ---------------- Gather (NHWC axis remap) ---------------- */
    case OpType::Gather: {
        if (n_ins < 2) return false;
        const Tensor& data = *ins_ptr[0];
        const Tensor& idx_t= *ins_ptr[1];
        const int64_t rank = data.rank();
        int64_t ax = node.params.axis;
        if (ax < 0) ax += rank;

        if (rank == 4) {
            if (ax == 1) ax = 3;
            else if (ax == 2) ax = 1;
            else if (ax == 3) ax = 2;
        }

        if (ax < 0 || ax >= rank) return false;

        const int64_t n_idx = idx_t.numel();

        int64_t out_dims[8];
        int64_t orank = 0;
        for (int64_t d = 0; d < ax; ++d) out_dims[orank++] = data.dim(d);
        out_dims[orank++] = n_idx;
        for (int64_t d = ax + 1; d < rank; ++d) out_dims[orank++] = data.dim(d);

        Shape os(out_dims, (int32_t)orank);
        Tensor out = acquire_output_buffer(os, data.dtype());
        if (out.is_empty()) return false;

        int64_t outer = 1;
        for (int64_t d = 0; d < ax; ++d) outer *= data.dim(d);
        int64_t inner = 1;
        for (int64_t d = ax + 1; d < rank; ++d) inner *= data.dim(d);
        const int64_t ax_dim = data.dim(ax);

        const float* dp = data.data<float>();
        float* op_ = out.data<float>();

        for (int64_t o = 0; o < outer; ++o) {
            for (int64_t j = 0; j < n_idx; ++j) {
                int64_t gi = get_tensor_int(&idx_t, j);
                if (gi < 0) gi += ax_dim;
                if (gi < 0 || gi >= ax_dim) gi = 0;
                const float* src = dp + (o * ax_dim + gi) * inner;
                float*       dst = op_ + (o * n_idx + j) * inner;
                std::memcpy(dst, src, sizeof(float) * (size_t)inner);
            }
        }

        const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
        if (out_idx >= 0) intermediate_[out_idx] = std::move(out);
        return true;
    }

    /* ---------------- Transpose ---------------- */
    case OpType::Transpose: {
        if (n_ins < 2) return false;
        const Tensor& data = *ins_ptr[0];
        const Tensor& perm_t= *ins_ptr[1];
        const int64_t rank = data.rank();
        if (perm_t.numel() != rank) return false;

        int64_t new_dims[8], perm_v[8];
        if (rank == 4 && data.dtype() == DType::F32) {
            int64_t onnx_perm[4];
            for (int d = 0; d < 4; ++d) {
                int64_t p = get_tensor_int(&perm_t, d);
                if (p < 0) p += 4;
                onnx_perm[d] = p;
            }

            /* ----------------------------------------------------------
             * NCHW↔NHWC permutation'ları motorda IDENTITY'dir.
             * Motor zaten fiziksel NHWC saklıyor; ONNX'in bu iki perm'i
             * sadece mantıksal layout etiketini değiştirir, fiziksel
             * veri sırasını değiştirmez (ConvNeXt LayerNorm girişi).
             * ---------------------------------------------------------- */
            const bool is_nchw2nhwc =
                (onnx_perm[0] == 0 && onnx_perm[1] == 2 &&
                 onnx_perm[2] == 3 && onnx_perm[3] == 1);
            const bool is_nhwc2nchw =
                (onnx_perm[0] == 0 && onnx_perm[1] == 3 &&
                 onnx_perm[2] == 1 && onnx_perm[3] == 2);

            if (is_nchw2nhwc || is_nhwc2nchw) {
                for (int64_t d = 0; d < rank; ++d) new_dims[d] = data.dim(d);
                Shape os(new_dims, (int32_t)rank);
                Tensor out = acquire_output_buffer(os, data.dtype());
                if (out.is_empty()) return false;
                std::memcpy(out.data(), data.data(), data.nbytes());
                const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
                if (out_idx >= 0)
                    intermediate_[static_cast<size_t>(out_idx)] = std::move(out);
                return true;
            }

            /* ONNX NCHW ekseni <-> Motor NHWC ekseni eşlemeleri */
            static const int64_t onnx2motor[4] = { 0, 3, 1, 2 };
            static const int64_t motor2onnx[4] = { 0, 2, 3, 1 };
            for (int d = 0; d < 4; ++d) {
                perm_v[d] = onnx2motor[ onnx_perm[ motor2onnx[d] ] ];
            }
        } else if (rank == 5 && data.dtype() == DType::F32) {
            /* CHANNEL SHUFFLE 5D: ONNX [N,g,C/g,H,W] ↔ motor [N,H,W,g,C/g]
             * onnx2motor5 = motor2onnx5 = {0,3,4,1,2} */
            int64_t onnx_perm[5];
            for (int d = 0; d < 5; ++d) {
                int64_t p = get_tensor_int(&perm_t, d);
                if (p < 0) p += 5;
                onnx_perm[d] = p;
            }
            static const int64_t onnx2motor5[5] = { 0, 3, 4, 1, 2 };
            static const int64_t motor2onnx5[5] = { 0, 3, 4, 1, 2 };
            for (int d = 0; d < 5; ++d) {
                perm_v[d] = onnx2motor5[ onnx_perm[ motor2onnx5[d] ] ];
            }
        } else {
            for (int64_t d = 0; d < rank; ++d) {
                int64_t p = get_tensor_int(&perm_t, d);
                if (p < 0) p += rank;
                perm_v[d] = p;
            }
        }
        for (int64_t d = 0; d < rank; ++d) new_dims[d] = data.dim(perm_v[d]);

        Shape os(new_dims, (int32_t)rank);
        Tensor out = acquire_output_buffer(os, data.dtype());
        if (out.is_empty()) return false;

        const float* dp = data.data<float>();
        float* op_ = out.data<float>();

        int64_t i1 = -1, i2 = -1, swaps = 0;
        for (int64_t d = 0; d < rank; ++d) {
            if (perm_v[d] != d) {
                if (swaps == 0)      i1 = d;
                else if (swaps == 1) i2 = d;
                swaps++;
            }
        }
        const bool is_two_dim_swap =
            (swaps == 2 && i1 >= 0 && i2 > i1 &&
             perm_v[i1] == i2 && perm_v[i2] == i1);

        if (is_two_dim_swap) {
            int64_t outer = 1;
            for (int64_t d = 0; d < i1; ++d)  outer *= data.dim(d);
            const int64_t A = data.dim(i1);
            int64_t mid = 1;
            for (int64_t d = i1 + 1; d < i2; ++d) mid *= data.dim(d);
            const int64_t B = data.dim(i2);
            int64_t inner = 1;
            for (int64_t d = i2 + 1; d < rank; ++d) inner *= data.dim(d);

            const int64_t surface = outer * A * mid * B;
            constexpr int64_t CHUNK = 1024;
            const int64_t k_chunks = (inner + CHUNK - 1) / CHUNK;

            auto& pool = runtime::GlobalPool::get();
            pool.parallel_for(0, surface * k_chunks, [&](int64_t t0, int64_t t1) {
                for (int64_t t = t0; t < t1; ++t) {
                    const int64_t kc = t % k_chunks;
                    const int64_t s  = t / k_chunks;

                    const int64_t k0 = kc * CHUNK;
                    const int64_t k_count = std::min<int64_t>(CHUNK, inner - k0);
                    const size_t bytes = static_cast<size_t>(k_count) * sizeof(float);

                    int64_t tmp = s;
                    const int64_t a = tmp % A;   tmp /= A;
                    const int64_t m = (mid > 1) ? (tmp % mid) : 0;
                    if (mid > 1) tmp /= mid;
                    const int64_t b = tmp % B;   tmp /= B;
                    const int64_t o = tmp;

                    const float* src = dp + ((((o * A + a) * mid + m) * B + b) * inner) + k0;
                    float*       dst = op_ + ((((o * B + b) * mid + m) * A + a) * inner) + k0;
                    std::memcpy(dst, src, bytes);
                }
            });
        } else {
            const int64_t n = out.numel();
            int64_t old_str[8];
            old_str[rank - 1] = 1;
            for (int64_t d = rank - 2; d >= 0; --d)
                old_str[d] = old_str[d + 1] * data.dim(d + 1);

            int64_t o_idx[8];
            for (int64_t i = 0; i < n; ++i) {
                int64_t t = i;
                for (int64_t d = rank - 1; d >= 0; --d) {
                    o_idx[d] = t % new_dims[d];
                    t /= new_dims[d];
                }
                int64_t src = 0;
                for (int64_t d = 0; d < rank; ++d) {
                    src += o_idx[d] * old_str[perm_v[d]];
                }
                op_[i] = dp[src];
            }
        }

        const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
        if (out_idx >= 0) intermediate_[out_idx] = std::move(out);
        return true;
    }

    /* ---------------- Expand ---------------- */
    case OpType::Expand: {
        if (n_ins < 2) return false;
        const Tensor& data = *ins_ptr[0];
        const Tensor& tshp = *ins_ptr[1];
        const int64_t rank = tshp.numel();
        if (rank < 1 || rank > 8) return false;

        int64_t new_dims[8];
        for (int64_t d = 0; d < rank; ++d) new_dims[d] = get_tensor_int(&tshp, d);

        Shape os(new_dims, (int32_t)rank);
        Tensor out = acquire_output_buffer(os, data.dtype());
        if (out.is_empty()) return false;

        const float* dp = data.data<float>();
        float* op_ = out.data<float>();
        const int64_t n = out.numel();
        const int64_t dn = data.numel();

        if (dn == n) {
            std::memcpy(op_, dp, sizeof(float) * (size_t)n);
        } else if (dn == 1) {
            const float v = dp[0];
            for (int64_t i = 0; i < n; ++i) op_[i] = v;
        } else {
            const int64_t dr = data.rank();
            int64_t o_idx[8];
            for (int64_t i = 0; i < n; ++i) {
                int64_t t = i;
                for (int64_t d = rank - 1; d >= 0; --d) {
                    o_idx[d] = t % new_dims[d];
                    t /= new_dims[d];
                }
                int64_t src = 0;
                for (int64_t d = 0; d < dr; ++d) {
                    const int64_t sd = (data.dim(d) > 1) ? o_idx[d] : 0;
                    src = src * data.dim(d) + sd;
                }
                op_[i] = dp[src];
            }
        }

        const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
        if (out_idx >= 0) intermediate_[out_idx] = std::move(out);
        return true;
    }

    /* ---------------- Unsqueeze ---------------- */
    case OpType::Unsqueeze: {
        if (n_ins == 0) return false;
        const Tensor& in = *ins_ptr[0];
        const int64_t rank = in.rank();

        std::vector<int64_t> axes;
        if (n_ins > 1 && !ins_ptr[1]->is_empty()) {
            for (int64_t i = 0; i < ins_ptr[1]->numel(); ++i)
                axes.push_back(get_tensor_int(ins_ptr[1], i));
        }
        if (axes.empty()) return false;

        const int64_t out_rank = rank + (int64_t)axes.size();
        if (out_rank > 8) return false;
        for (auto& a : axes) {
            if (a < 0) a += out_rank;
        }
        std::sort(axes.begin(), axes.end());

        int64_t new_dims[8];
        int64_t di = 0, ai = 0;
        for (int64_t d = 0; d < out_rank; ++d) {
            if (ai < (int64_t)axes.size() && axes[ai] == d) {
                new_dims[d] = 1;
                ++ai;
            } else {
                new_dims[d] = in.dim(di);
                ++di;
            }
        }
        Shape os(new_dims, (int32_t)out_rank);
        Tensor out = acquire_output_buffer(os, in.dtype());
        if (out.is_empty()) return false;
        std::memcpy(out.data(), in.data(), in.nbytes());
        const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
        if (out_idx >= 0) intermediate_[out_idx] = std::move(out);
        return true;
    }

    /* ---------------- Squeeze ---------------- */
    case OpType::Squeeze: {
        if (n_ins == 0) return false;
        const Tensor& in = *ins_ptr[0];
        const int64_t rank = in.rank();

        std::vector<int64_t> axes;
        if (n_ins > 1 && !ins_ptr[1]->is_empty()) {
            for (int64_t i = 0; i < ins_ptr[1]->numel(); ++i) {
                int64_t a = get_tensor_int(ins_ptr[1], i);
                if (a < 0) a += rank;
                axes.push_back(a);
            }
        }
        if (axes.empty()) {
            for (int64_t d = 0; d < rank; ++d)
                if (in.dim(d) == 1) axes.push_back(d);
        }
        std::sort(axes.begin(), axes.end());
        axes.erase(std::unique(axes.begin(), axes.end()), axes.end());

        int64_t new_dims[8];
        int64_t nr = 0;
        int64_t ai = 0;
        for (int64_t d = 0; d < rank; ++d) {
            if (ai < (int64_t)axes.size() && axes[ai] == d) {
                ++ai;
                continue;
            }
            new_dims[nr++] = in.dim(d);
        }
        if (nr == 0) { new_dims[0] = 1; nr = 1; }

        Shape os(new_dims, (int32_t)nr);
        Tensor out = acquire_output_buffer(os, in.dtype());
        if (out.is_empty()) return false;
        std::memcpy(out.data(), in.data(), in.nbytes());
        const int32_t out_idx = node.outputs.empty() ? 0 : node.outputs[0];
        if (out_idx >= 0) intermediate_[out_idx] = std::move(out);
        return true;
    }