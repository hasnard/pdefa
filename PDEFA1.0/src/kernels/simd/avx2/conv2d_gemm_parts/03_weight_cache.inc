/* ============================================================================
 *  03_weight_cache.inc — NHWC weight packing
 *  PyTorch weight: [Cout][Cin][KH][KW]  (NCHW)
 *  Packed output : [K][Cout_padded]     (K = (kh*KW + kw)*Cin + ci)
 * ========================================================================== */

std::shared_mutex g_wcache_mtx;
std::unordered_map<const float*, std::vector<float>> g_wcache;

const std::vector<float>* get_packed_weight_16(const float* weight,
                                               int64_t K, int64_t Cout,
                                               int64_t Cin, int64_t KH, int64_t KW) noexcept
{
    {
        std::shared_lock<std::shared_mutex> read_lock(g_wcache_mtx);
        auto it = g_wcache.find(weight);
        if (it != g_wcache.end()) return &it->second;
    }

    std::unique_lock<std::shared_mutex> write_lock(g_wcache_mtx);
    auto it = g_wcache.find(weight);
    if (it != g_wcache.end()) return &it->second;

    const int64_t cout_padded = ((Cout + 15) / 16) * 16;
    std::vector<float> packed(static_cast<size_t>(cout_padded) * static_cast<size_t>(K), 0.0f);

    /* K indexing: k = (kh*KW + kw)*Cin + ci */
    for (int64_t kh = 0; kh < KH; ++kh) {
        for (int64_t kw = 0; kw < KW; ++kw) {
            for (int64_t ci = 0; ci < Cin; ++ci) {
                const int64_t k = (kh * KW + kw) * Cin + ci;
                for (int64_t n = 0; n < Cout; n += 16) {
                    const int64_t nr = std::min<int64_t>(16, Cout - n);
                    float* dst = packed.data() + n * K + k * 16;
                    for (int64_t i = 0; i < nr; ++i) {
                        /* PyTorch: weight[co][ci][kh][kw] */
                        dst[i] = weight[((n + i) * Cin + ci) * KH * KW + kh * KW + kw];
                    }
                }
            }
        }
    }

    auto [new_it, inserted] = g_wcache.emplace(weight, std::move(packed));
    return &new_it->second;
}