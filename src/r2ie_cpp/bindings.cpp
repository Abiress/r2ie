// bindings.cpp
// pybind11 bindings exposing the C++ HDQMemoryManager to Python as r2ie_cpp.HDQMemory.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "hdq_memory.hpp"

namespace py = pybind11;
using r2ie::HDQMemoryManager;
using r2ie::MassState;

PYBIND11_MODULE(_r2ie_cpp, m) {
    m.doc() = "C++ HDQ mass-memory manager for R2IE (double-buffered, lock-free swap).";

    py::class_<MassState>(m, "MassState")
        .def("norm", &MassState::norm);

    py::class_<HDQMemoryManager>(m, "HDQMemory")
        .def(py::init<size_t, float, float>(), py::arg("size"),
             py::arg("decay") = 0.95f, py::arg("clamp") = 5.0f)
        .def("apply_accretion", &HDQMemoryManager::apply_accretion,
             py::arg("delta"))
        .def("active_norm", &HDQMemoryManager::active_norm)
        .def("weights",
             [](const HDQMemoryManager& m) {
                 const auto* s = m.get_read_state();
                 return py::array_t<float>(s->weights.size(), s->weights.data());
             })
        .def("reset", &HDQMemoryManager::reset)
        .def("dim", &HDQMemoryManager::dim)
        .def("decay", &HDQMemoryManager::decay)
        .def("clamp", &HDQMemoryManager::clamp);
}
