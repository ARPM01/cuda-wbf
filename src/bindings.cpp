#include "core.hpp"
#ifdef WBF_HAS_CUDA
#include "cuda.hpp"
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <limits>

namespace py = pybind11;

namespace {
// These aliases describe NumPy arrays whose values are laid out contiguously
// in memory. A candidate row occupies five consecutive float64 values.
using Rows = py::array_t<double, py::array::c_style>;
using Offsets = py::array_t<std::int64_t, py::array::c_style>;

void validate_inputs(const Rows& candidates, const Offsets& offsets,
                     std::size_t sources, double weight_sum, double iou_thr) {
    if (candidates.ndim() != 2 || candidates.shape(1) != 5) {
        throw py::value_error("candidates must have shape (N, 5)");
    }
    // Offsets are row indices. Validate the boundaries before the core uses
    // them to access the NumPy buffer. Strict increases ensure nonempty groups.
    if (offsets.ndim() != 1 || offsets.size() < 1 || offsets.data()[0] != 0 ||
        offsets.data()[offsets.size() - 1] != candidates.shape(0)) {
        throw py::value_error("offsets must start at zero and end at the candidate count");
    }
    for (py::ssize_t i = 1; i < offsets.size(); ++i) {
        if (offsets.data()[i] <= offsets.data()[i - 1] || offsets.data()[i] > candidates.shape(0)) {
            throw py::value_error("group offsets must be strictly increasing and within the candidate array");
        }
    }
    if (sources == 0 || !std::isfinite(weight_sum) || weight_sum <= 0 ||
        !std::isfinite(iou_thr) || iou_thr < 0 || iou_thr > 1) {
        throw py::value_error("invalid source count, weight sum, or IoU threshold");
    }
    for (py::ssize_t i = 0; i < candidates.shape(0); ++i) {
        const auto* row = candidates.data() + i * 5;
        for (std::size_t j = 0; j < 5; ++j) {
            if (!std::isfinite(row[j]) || (j > 0 && (row[j] < 0 || row[j] > 1))) {
                throw py::value_error("candidate scores and normalized coordinates must be finite");
            }
        }
        if (row[0] <= 0 || row[3] <= row[1] || row[4] <= row[2]) {
            throw py::value_error("candidates must have positive scores and box area");
        }
    }
}

py::tuple numpy_result(const wbf::Result& result) {
    if (result.rows.size() > static_cast<std::size_t>(std::numeric_limits<py::ssize_t>::max() / 5)) {
        throw std::overflow_error("fusion result is too large for a NumPy array");
    }
    // The core returns C++ vectors. Copy them into NumPy arrays that own
    // their memory so the returned data remains valid after this call ends.
    Rows values({static_cast<py::ssize_t>(result.rows.size()), py::ssize_t{5}});
    auto output = values.mutable_unchecked<2>();
    for (std::size_t i = 0; i < result.rows.size(); ++i) {
        for (std::size_t j = 0; j < 5; ++j) {
            output(i, j) = result.rows[i][j];
        }
    }
    Offsets output_offsets(static_cast<py::ssize_t>(result.offsets.size()));
    auto offset_view = output_offsets.mutable_unchecked<1>();
    for (std::size_t i = 0; i < result.offsets.size(); ++i) {
        offset_view(i) = result.offsets[i];
    }
    return py::make_tuple(values, output_offsets);
}

py::tuple fuse_groups(const Rows& candidates, const Offsets& offsets,
                      std::size_t sources, double weight_sum, double iou_thr) {
    validate_inputs(candidates, offsets, sources, weight_sum, iou_thr);
    const auto* data = candidates.data();
    const auto* slices = offsets.data();
    const auto count = static_cast<std::size_t>(candidates.shape(0));
    const auto groups = static_cast<std::size_t>(offsets.size() - 1);
    wbf::Result result;
    {
        // Other Python threads can run during fusion. Leaving this block
        // reacquires the interpreter lock before numpy_result creates arrays.
        py::gil_scoped_release release;
        result = wbf::fuse_groups(data, count, slices, groups, sources, weight_sum, iou_thr);
    }
    return numpy_result(result);
}
}  // namespace

PYBIND11_MODULE(_native, module) {
    module.doc() = "CUDA WBF native extension";
    module.def("build_info", [] {
        py::dict info;
        info["cxx_standard"] = 17;
#ifdef WBF_HAS_CUDA
        info["cuda_enabled"] = true;
#else
        info["cuda_enabled"] = false;
#endif
        return info;
    });
    // Register the wrapper as _native.fuse_groups. noconvert() requires callers
    // to supply arrays with the expected dtype and layout at the boundary.
    module.def("fuse_groups", &fuse_groups, py::arg("candidates").noconvert(),
               py::arg("offsets").noconvert(), py::arg("sources"),
               py::arg("weight_sum"), py::arg("iou_thr"));
    module.def("available_backends", [] {
        std::vector<std::string> backends{"cpp"};
#ifdef WBF_HAS_CUDA
        backends.push_back("cuda_groups");
        backends.push_back("cuda_coop");
#endif
        return backends;
    });
#ifdef WBF_HAS_CUDA
    py::class_<wbf::CudaWorkspace>(module, "CudaWorkspace")
        .def(py::init<>())
        .def("upload", [](wbf::CudaWorkspace& workspace, const Rows& candidates, const Offsets& offsets,
                           std::size_t sources, double weight_sum, double iou_thr) {
            validate_inputs(candidates, offsets, sources, weight_sum, iou_thr);
            const auto* data = candidates.data();
            const auto* slices = offsets.data();
            const auto count = static_cast<std::size_t>(candidates.shape(0));
            const auto groups = static_cast<std::size_t>(offsets.size() - 1);
            py::gil_scoped_release release;
            workspace.upload(data, count, slices, groups, sources, weight_sum, iou_thr);
        }, py::arg("candidates").noconvert(), py::arg("offsets").noconvert(),
           py::arg("sources"), py::arg("weight_sum"), py::arg("iou_thr"))
        .def("launch", &wbf::CudaWorkspace::launch, py::arg("backend"), py::call_guard<py::gil_scoped_release>())
        .def("download", [](wbf::CudaWorkspace& workspace) {
            wbf::Result result;
            { py::gil_scoped_release release; result = workspace.download(); }
            return numpy_result(result);
        })
        .def("fuse", [](wbf::CudaWorkspace& workspace, const Rows& candidates, const Offsets& offsets,
                         std::size_t sources, double weight_sum, double iou_thr, const std::string& backend) {
            validate_inputs(candidates, offsets, sources, weight_sum, iou_thr);
            const auto* data = candidates.data();
            const auto* slices = offsets.data();
            const auto count = static_cast<std::size_t>(candidates.shape(0));
            const auto groups = static_cast<std::size_t>(offsets.size() - 1);
            wbf::Result result;
            {
                py::gil_scoped_release release;
                result = workspace.fuse(data, count, slices, groups, sources, weight_sum, iou_thr, backend);
            }
            return numpy_result(result);
        }, py::arg("candidates").noconvert(), py::arg("offsets").noconvert(),
           py::arg("sources"), py::arg("weight_sum"), py::arg("iou_thr"), py::arg("backend"))
        .def_property_readonly("stats", [](const wbf::CudaWorkspace& workspace) {
            const auto stats = workspace.stats();
            py::dict result;
            result["capacity_boxes"] = stats.capacity_boxes;
            result["capacity_groups"] = stats.capacity_groups;
            result["allocated_bytes"] = stats.allocated_bytes;
            result["upload_bytes"] = stats.upload_bytes;
            result["download_bytes"] = stats.download_bytes;
            result["input_boxes"] = stats.input_boxes;
            result["groups"] = stats.groups;
            result["kernel_launches"] = stats.kernel_launches;
            result["device"] = stats.device;
            return result;
        });
#endif
}
