# pdefa — CPU inference engine with built-in image loader + YOLO post-process.
# Copyright (C) 2026 Hasan Arda Acar
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Commercial license available: info@pdfduty.com
"""
pdefa — CPU inference engine with built-in image loader + YOLO post-process.

Example:
    import pdefa

    # 1) Load model
    ctx     = pdefa.Context()
    model   = pdefa.Model(ctx, "yolov8n.engine")
    session = pdefa.Session(ctx, model)

    # 2) Prepare image
    tensor, meta = pdefa.prepare_for_engine("earth.jpg", 640, 640)

    # 3) Inference
    session.set_input(0, tensor)
    session.run()
    out = session.output(0)              # Tensor
    arr = out.tolist()                    # nested list (no numpy)

    # 4) Map box back to original coordinates
    b = pdefa.map_box_to_original(pdefa.Box(100, 200, 300, 400), meta)
    print(b)
"""
__version__ = "1.0.3"
__license__ = "AGPL-3.0-or-later"

from ._bindings import (
    # --- Meta / types ---
    PdefaError,
    PreprocessMeta,
    Box,
    Image,
    Tensor,
    TensorInfo,
    Context,
    Model,
    Session,

    # --- Version / CPU ---
    version,
    simd_level,
    cpu_brand,

    # --- Image / Preprocess ---
    load_image,
    preprocess_letterbox,
    prepare_for_engine,
    load_image_as_tensor,

    # --- Post-process ---
    map_box_to_original,
    map_point_to_original,
    decode_yolo,
    prepare_for_model,
    _model_input_shape,

    # --- Legacy ops ---
    ops_matmul, ops_relu, ops_silu,
    tensor_empty,

    # --- Binary ---
    ops_add, ops_sub, ops_mul, ops_div,

    # --- Unary ---
    ops_sigmoid, ops_tanh, ops_gelu, ops_clip, ops_softmax,

    # --- Reduce ---
    ops_reduce_max, ops_reduce_sum, ops_reduce_mean,

    # --- Index / Shape ---
    ops_argmax, ops_gather,
    ops_reshape, ops_transpose, ops_concat, ops_stack,

    # --- Random ---
    random_randn, random_rand,

    # --- Tensor creation without numpy ---
    randn, rand, zeros, ones, empty, from_list,

    # --- Convenience API ---
    load, EngineModel,
)

from .batch import BatchInference

__all__ = [
    "__version__",

    # Types
    "PdefaError", "PreprocessMeta", "Box", "Image", "Tensor", "TensorInfo",
    "Context", "Model", "Session",

    # Meta
    "version", "simd_level", "cpu_brand",

    # Image / Preprocess
    "load_image", "preprocess_letterbox", "prepare_for_engine",
    "load_image_as_tensor",

    # Post-process
    "map_box_to_original", "map_point_to_original",
    "decode_yolo", "prepare_for_model",

    # Ops
    "ops_matmul", "ops_relu", "ops_silu", "tensor_empty",
    "ops_add", "ops_sub", "ops_mul", "ops_div",
    "ops_sigmoid", "ops_tanh", "ops_gelu", "ops_clip", "ops_softmax",
    "ops_reduce_max", "ops_reduce_sum", "ops_reduce_mean",
    "ops_argmax", "ops_gather",
    "ops_reshape", "ops_transpose", "ops_concat", "ops_stack",
    "random_randn", "random_rand",

    # Tensor creation without numpy
    "randn", "rand", "zeros", "ones", "empty", "from_list",

    # Convenience API
    "load", "EngineModel",

    # Batch
    "BatchInference",
]