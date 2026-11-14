import cuda_wbf


def test_native_extension_loads():
    info = cuda_wbf.build_info()
    assert info["cxx_standard"] == 17
    assert isinstance(info["cuda_enabled"], bool)
    backends = cuda_wbf.available_backends()
    assert "cpp" in backends
    assert ("cuda_groups" in backends) == info["cuda_enabled"]
