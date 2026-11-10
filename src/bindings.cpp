#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_native, module) {
    module.doc() = "CUDA WBF native extension";
    module.def("build_info", [] {
        py::dict info;
        info["cxx_standard"] = 17;
        info["cuda_enabled"] = false;
        return info;
    });
}
