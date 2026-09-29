/* ============================================================
 *  Anonim Yardimcilar
 *  (session.cpp icinde "namespace { ... }" blogu icine include edilir)
 * ============================================================ */

const Tensor* resolve_tensor(const Model& m, int32_t idx,
                             const std::vector<const Tensor*>& ext,
                             const std::vector<Tensor>& inter) noexcept
{
    if (idx >= 0) {
        const size_t i = static_cast<size_t>(idx);
        if (i < inter.size() && !inter[i].is_empty()) return &inter[i];
        if (i < ext.size()   && ext[i] != nullptr)    return ext[i];
        return nullptr;
    }
    else if (idx <= -1000) {
        const size_t w_idx = static_cast<size_t>(-idx - 1000);
        return (w_idx < static_cast<size_t>(m.num_weights()))
                   ? &m.weight(static_cast<int32_t>(w_idx)) : nullptr;
    }
    else {
        const size_t ext_idx = static_cast<size_t>(-idx - 1);
        return (ext_idx < ext.size()) ? ext[ext_idx] : nullptr;
    }
}

const char* op_type_name(OpType op) noexcept {
    switch (op) {
        case OpType::Conv2d:      return "Conv2d";
        case OpType::MatMul:      return "MatMul";
        case OpType::ReLU:        return "ReLU";
        case OpType::LeakyReLU:   return "LeakyReLU";
        case OpType::Sigmoid:     return "Sigmoid";
        case OpType::Tanh:        return "Tanh";
        case OpType::SiLU:        return "SiLU";
        case OpType::GELU:        return "GELU";
        case OpType::MaxPool2d:   return "MaxPool2d";
        case OpType::AvgPool2d:   return "AvgPool2d";
        case OpType::BatchNorm2d: return "BatchNorm2d";
        case OpType::Concat:      return "Concat";
        case OpType::Add:         return "Add";
        case OpType::Mul:         return "Mul";
        case OpType::Sub:         return "Sub";
        case OpType::Div:         return "Div";
        case OpType::Transpose:   return "Transpose";
        case OpType::Slice:       return "Slice";
        case OpType::Gather:      return "Gather";
        case OpType::Shape:       return "Shape";
        case OpType::Expand:      return "Expand";
        case OpType::Upsample:    return "Upsample";
        case OpType::Resize:      return "Resize";
        case OpType::Reshape:     return "Reshape";
        case OpType::Unsqueeze:   return "Unsqueeze";
        case OpType::Squeeze:     return "Squeeze";
        case OpType::TopK:            return "TopK";
        case OpType::GatherElements:  return "GatherElements";
        case OpType::Mod:             return "Mod";
        case OpType::ReduceMax:       return "ReduceMax";
        case OpType::ReduceMin:       return "ReduceMin";
        case OpType::ReduceProd:      return "ReduceProd";
        case OpType::ReduceSum:       return "ReduceSum";
        case OpType::NonZero:         return "NonZero";
        case OpType::ConstantOfShape: return "ConstantOfShape";
        case OpType::Tile:            return "Tile";
        case OpType::Range:           return "Range";
        case OpType::Equal:           return "Equal";
        case OpType::Greater:         return "Greater";
        case OpType::Exp:             return "Exp";
        case OpType::Min:             return "Min";
        case OpType::Max:             return "Max";
        case OpType::Softmax:     return "Softmax";
        case OpType::Input:       return "Input";
        case OpType::Output:      return "Output";
        default:                  return "Unknown";
    }
}

/* Profiling #ifdef ENGINE_PROFILING arkasinda.
 * Release build'de hic derlenmez, sifir overhead. */
FILE* get_profile_csv() noexcept {
#ifdef ENGINE_PROFILING
    static FILE* g_csv     = nullptr;
    static bool  g_checked = false;
    if (!g_checked) {
        g_checked = true;
        const char* path = std::getenv("ENGINE_PROFILE_CSV");
        if (path && *path) {
            g_csv = std::fopen(path, "w");
            if (g_csv) {
                std::fprintf(g_csv, "node_idx,op_type,dims,time_ms\n");
                std::fflush(g_csv);
            }
        }
    }
    return g_csv;
#else
    return nullptr;
#endif
}




/* ============================================================================
 *  AVX2 softmax yardımcıları
 * ========================================================================== */
inline __m256 softmax_exp_avx2(__m256 x) noexcept {
    const __m256 kHi = _mm256_set1_ps(88.3762626647949f);
    const __m256 kLo = _mm256_set1_ps(-88.3762626647949f);
    x = _mm256_min_ps(x, kHi);
    x = _mm256_max_ps(x, kLo);

    const __m256 kLog2e = _mm256_set1_ps(1.44269504088896341f);
    __m256 fx = _mm256_mul_ps(x, kLog2e);
    __m256 n  = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    const __m256 kC1 = _mm256_set1_ps(0.693359375f);
    const __m256 kC2 = _mm256_set1_ps(-2.121944400546905827679e-4f);
    __m256 y = _mm256_fnmadd_ps(n, kC1, x);
    y = _mm256_fnmadd_ps(n, kC2, y);

    const __m256 c0 = _mm256_set1_ps(1.9875691500e-4f);
    const __m256 c1 = _mm256_set1_ps(1.3981999507e-3f);
    const __m256 c2 = _mm256_set1_ps(8.3334519073e-3f);
    const __m256 c3 = _mm256_set1_ps(4.1665795894e-2f);
    const __m256 c4 = _mm256_set1_ps(1.6666665459e-1f);
    const __m256 c5 = _mm256_set1_ps(5.0000001201e-1f);

    __m256 p = _mm256_fmadd_ps(c0, y, c1);
    p = _mm256_fmadd_ps(p, y, c2);
    p = _mm256_fmadd_ps(p, y, c3);
    p = _mm256_fmadd_ps(p, y, c4);
    p = _mm256_fmadd_ps(p, y, c5);
    __m256 y2 = _mm256_mul_ps(y, y);
    p = _mm256_fmadd_ps(p, y2, y);
    p = _mm256_add_ps(p, _mm256_set1_ps(1.0f));

    __m256i n_int = _mm256_cvtps_epi32(n);
    __m256 scale = _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_add_epi32(n_int, _mm256_set1_epi32(127)), 23));
    return _mm256_mul_ps(p, scale);
}

inline __m256 softmax_rcp_avx2(__m256 d) noexcept {
    const __m256 two = _mm256_set1_ps(2.0f);
    __m256 r = _mm256_rcp_ps(d);
    return _mm256_mul_ps(r, _mm256_fnmadd_ps(d, r, two));
}




/* ============================================================================
 *  DType Duyarlı Tamsayı Okuyucu
 *  ONNX metadata tensörlerindeki (I64, I32, F32) tip uyuşmazlıklarını çözer.
 * ========================================================================== */
inline int64_t get_tensor_int(const Tensor* t, int64_t idx) {
    if (!t || idx < 0 || idx >= t->numel()) return 0;
    switch (t->dtype()) {
        case DType::I64:
            return t->data<int64_t>()[idx];
        case DType::I32:
            return static_cast<int64_t>(t->data<int32_t>()[idx]);
        case DType::F32:
        default:
            /* ONNX Slice/Reshape starts/ends float gelirse TRUNCATE et (int cast).
             * Örnek: (C+1)/2 = 32.5 → int(32.5) = 32 ✓ (round olsaydı 33 ✗).
             * Bu, YOLO C2f chunk(2, dim=1) split'i doğru yapar. */
            return static_cast<int64_t>(t->data<float>()[idx]);
    }
}



inline float get_tensor_float(const Tensor* t, int64_t idx) {
    if (!t || idx < 0 || idx >= t->numel()) return 0.0f;
    switch (t->dtype()) {
        case DType::F32:
            return t->data<float>()[idx];
        case DType::I64:
            return static_cast<float>(t->data<int64_t>()[idx]);
        case DType::I32:
            return static_cast<float>(t->data<int32_t>()[idx]);
        default:
            return 0.0f;
    }
}