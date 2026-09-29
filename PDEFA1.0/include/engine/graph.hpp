/* ============================================================================
 *  include/engine/graph.hpp
 *  Engine-AI — Basit computation graph
 *
 *  Model bir DAG (yönlü asiklik graf) olarak temsil edilir.
 *  Her Node bir op (conv, matmul, relu...), her edge bir tensor.
 * ========================================================================== */

#ifndef ENGINE_GRAPH_HPP
#define ENGINE_GRAPH_HPP

#include "engine/tensor.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace engine {

/* Op tipleri — loader bunu okur, session dispatch eder */
enum class OpType : uint8_t {
    Unknown      = 0,
    Conv2d       = 1,
    MatMul       = 2,
    Add          = 3,
    Mul          = 4,
    Sub          = 5,
    Div          = 6,
    ReLU         = 10,
    LeakyReLU    = 11,
    Sigmoid      = 12,
    Tanh         = 13,
    SiLU         = 14,
    GELU         = 15,
    MaxPool2d    = 20,
    AvgPool2d    = 21,
    BatchNorm2d  = 22,
    LayerNorm    = 23,
    Softmax      = 30,
    Concat       = 31,
    Upsample     = 32,
    Resize       = 33,
    Input        = 40,
    Output       = 41,
    Reshape      = 42,
    Transpose    = 43,
    Slice        = 44,
    Gather       = 45,
    Shape        = 46,
    Expand       = 47,
    Unsqueeze    = 48,
    HardSwish    = 16,   
    HardSigmoid  = 17, 
    Squeeze      = 49,
    TopK             = 50,
    GatherElements   = 51,
    Mod              = 52,
    ReduceMax        = 53,
    ReduceMin        = 54,
    ReduceProd       = 55,
    NonZero          = 56,
    ConstantOfShape  = 57,
    Tile             = 58,
    Range            = 59,
    Equal            = 60,
    Greater          = 61,
    Exp              = 62,
    Min              = 63,
    Max              = 64,
    ReduceSum        = 65,
    Clip             = 66,
    ArgMax           = 67,
    Stack            = 68,


};

struct OpParams {
    int64_t stride_h = 1, stride_w = 1;
    int64_t pad_h = 0, pad_w = 0;
    int64_t dil_h = 1, dil_w = 1;
    int64_t groups = 1;
    int64_t kernel_h = 2, kernel_w = 2;
    int32_t axis = -1;
    int64_t scale_h = 2, scale_w = 2;
    int64_t target_h = 0, target_w = 0;
    uint8_t upsample_mode = 0;
    float eps = 1e-5f;
    uint8_t fused_activation = 0;
    float leaky_slope = 0.01f;
    /* ONNX reduce ops (ReduceSum/Max/Min/Prod/Mean) — attribute.
     * -1 = attribute yok (bilinmiyor); 0 veya 1 = explicit değer.
     * Loader doldurur; doldurmazsa sezgisel devreye girer. */

};

struct Node {
    OpType             op = OpType::Unknown;
    std::string        name;
    std::vector<int32_t> inputs;
    std::vector<int32_t> outputs;
    OpParams           params;
};

/* ============================================================================
 *  YENİ (v2): Tensor metadata — input/output shape ve dtype
 *
 *  Dinamik boyutlar için -1 kullanılır (örn. batch size).
 * ========================================================================== */
struct TensorInfo {
    std::string name;                       // onnx ismi (opsiyonel)
    int64_t     shape[kMaxNdim] = {0};      // boyutlar
    int32_t     ndim = 0;                   // rank
    DType       dtype = DType::F32;         // veri tipi
};

/* Graph: node listesi + metadata */
struct Graph {
    std::vector<Node>    nodes;
    std::vector<int32_t> graph_inputs;
    std::vector<int32_t> graph_outputs;

    /* YENİ (v2): graph_inputs ve graph_outputs ile paralel sıralı */
    std::vector<TensorInfo> input_infos;
    std::vector<TensorInfo> output_infos;
};

} /* namespace engine */
#endif