/* ============================================================================
 *  08_weight_kmn.inc — Weight [Cout][Cin][KH][KW] → [K][Cout] (matmul için)
 *  K indexing: k = (kh*KW + kw)*Cin + ci
 * ========================================================================== */
std::shared_mutex g_wkmn_mtx;
std::unordered_map<const float*, std::vector<float>> g_wkmn;

const float* get_weight_kmn(const float* weight, int64_t K, int64_t Cout,
                            int64_t Cin, int64_t KH, int64_t KW) noexcept
{
    {
        std::shared_lock<std::shared_mutex> rl(g_wkmn_mtx);
        auto it = g_wkmn.find(weight);
        if (it != g_wkmn.end()) return it->second.data();
    }
    std::unique_lock<std::shared_mutex> wl(g_wkmn_mtx);
    auto it = g_wkmn.find(weight);
    if (it != g_wkmn.end()) return it->second.data();

    std::vector<float> kmn(static_cast<size_t>(K) * static_cast<size_t>(Cout));
    for (int64_t kh = 0; kh < KH; ++kh) {
        for (int64_t kw = 0; kw < KW; ++kw) {
            for (int64_t ci = 0; ci < Cin; ++ci) {
                const int64_t k = (kh * KW + kw) * Cin + ci;
                for (int64_t co = 0; co < Cout; ++co) {
                    kmn[k * Cout + co] =
                        weight[((co * Cin + ci) * KH + kh) * KW + kw];
                }
            }
        }
    }
    auto [new_it, _] = g_wkmn.emplace(weight, std::move(kmn));
    return new_it->second.data();
}