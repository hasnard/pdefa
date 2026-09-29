# pdefa - AGPL-3.0-or-later - see LICENSE
import re
import pdefa


def test_version_format():
    # Hardcoded değil, format kontrol et
    v = pdefa.version()
    assert isinstance(v, str)
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:[.\-][A-Za-z0-9]+)?", v), v


def test_imports():
    for name in (
        "Context", "Model", "Session", "Tensor", "Image",
        "load_image", "decode_yolo", "prepare_for_engine",
        "prepare_for_model", "preprocess_letterbox",
        "ops_add", "ops_matmul", "randn", "zeros", "ones",
        "BatchInference", "Box", "PreprocessMeta",
    ):
        assert hasattr(pdefa, name), f"missing export: {name}"


def test_simd():
    assert pdefa.simd_level() in ("AVX2", "AVX-512", "SSE", "SCALAR")


def test_cpu_brand():
    b = pdefa.cpu_brand()
    assert isinstance(b, str)
    assert len(b) > 0


def test_license_attribute():
    assert pdefa.__license__ == "AGPL-3.0-or-later"