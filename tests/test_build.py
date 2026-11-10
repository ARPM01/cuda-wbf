import cuda_wbf


def test_native_extension_loads():
    assert cuda_wbf.build_info() == {"cxx_standard": 17, "cuda_enabled": False}
