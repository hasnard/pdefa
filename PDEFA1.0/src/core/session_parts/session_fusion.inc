/* ============================================================
 *  Conv + Aktivasyon fusion
 *
 *  Iki kritik kontrol var:
 *    1. Conv cikisi tek tuketici mi? (coklu tuketici varsa fuse etme)
 *    2. Sigmoid cikisi tek tuketici mi? (SiLU pattern icin)
 *  Aksi halde graph bozulur: ikinci tuketici artik uretilmeyen
 *  bir tensor'u okumaya calisir.
 * ============================================================ */
void Session::fuse_conv_activations_() noexcept {
    Graph& g = const_cast<Graph&>(model_->graph());

    for (size_t i = 0; i < g.nodes.size(); ++i) {
        Node& conv = g.nodes[i];
        if (conv.op != OpType::Conv2d) continue;

        int32_t conv_out = conv.outputs.empty() ? -1 : conv.outputs[0];
        if (conv_out < 0) continue;

        /* Conv cikisinin kac tuketicisi var? */
        int conv_consumers = 0;
        for (size_t k = 0; k < g.nodes.size(); ++k) {
            if (k == i) continue;
            for (int32_t in : g.nodes[k].inputs) {
                if (in == conv_out) ++conv_consumers;
            }
        }
        for (int32_t go : g.graph_outputs) {
            if (go == conv_out) ++conv_consumers;
        }

        /* 1. Kol: SiLU Kontrolü (Conv cikisini hem Sigmoid hem Mul tuketir: 2 tüketici) */
        if (conv_consumers == 2) {
            int sig_idx = -1, mul_idx = -1;

            for (size_t j = i + 1; j < g.nodes.size(); ++j) {
                if (g.nodes[j].op == OpType::Sigmoid &&
                    g.nodes[j].inputs.size() == 1 &&
                    g.nodes[j].inputs[0] == conv_out) {

                    /* Sigmoid cikisi tek tuketici mi? */
                    const int32_t so = g.nodes[j].outputs.empty() ? -1 : g.nodes[j].outputs[0];
                    if (so < 0) break;

                    int sig_consumers = 0;
                    for (size_t k = 0; k < g.nodes.size(); ++k) {
                        if (k == j) continue;
                        for (int32_t in : g.nodes[k].inputs) {
                            if (in == so) ++sig_consumers;
                        }
                    }
                    for (int32_t go : g.graph_outputs) {
                        if (go == so) ++sig_consumers;
                    }
                    if (sig_consumers != 1) break;

                    sig_idx = static_cast<int>(j);
                    break;
                }
            }

            if (sig_idx != -1) {
                const int32_t sig_out = g.nodes[static_cast<size_t>(sig_idx)].outputs[0];
                for (size_t j = i + 1; j < g.nodes.size(); ++j) {
                    if (g.nodes[j].op == OpType::Mul && g.nodes[j].inputs.size() == 2) {
                        if ((g.nodes[j].inputs[0] == conv_out && g.nodes[j].inputs[1] == sig_out) ||
                            (g.nodes[j].inputs[1] == conv_out && g.nodes[j].inputs[0] == sig_out)) {
                            mul_idx = static_cast<int>(j);
                            break;
                        }
                    }
                }
            }

            if (sig_idx != -1 && mul_idx != -1) {
                conv.params.fused_activation = 5;
                conv.outputs[0] = g.nodes[static_cast<size_t>(mul_idx)].outputs[0];
                g.nodes[static_cast<size_t>(sig_idx)].op = OpType::Input;
                g.nodes[static_cast<size_t>(mul_idx)].op = OpType::Input;
                continue;
            }
        }

        /* 2. Kol: Tek tuketicili aktivasyonlar (ReLU, LeakyReLU, Sigmoid, Tanh) */
        if (conv_consumers != 1) continue;

        for (size_t j = i + 1; j < g.nodes.size(); ++j) {
            if (g.nodes[j].inputs.size() == 1 && g.nodes[j].inputs[0] == conv_out) {
                if (g.nodes[j].op == OpType::ReLU || g.nodes[j].op == OpType::LeakyReLU ||
                    g.nodes[j].op == OpType::Sigmoid || g.nodes[j].op == OpType::Tanh) {

                    if (g.nodes[j].op == OpType::ReLU) {
                        conv.params.fused_activation = 1;
                    } else if (g.nodes[j].op == OpType::LeakyReLU) {
                        conv.params.fused_activation = 2;
                        conv.params.leaky_slope = g.nodes[j].params.leaky_slope;
                    } else if (g.nodes[j].op == OpType::Sigmoid) {
                        conv.params.fused_activation = 3;
                    } else if (g.nodes[j].op == OpType::Tanh) {
                        conv.params.fused_activation = 4;
                    }
                    conv.outputs[0] = g.nodes[j].outputs[0];
                    g.nodes[j].op = OpType::Input;
                    break;
                }
            }
        }
    }
}