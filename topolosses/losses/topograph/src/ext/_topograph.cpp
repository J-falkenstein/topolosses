#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "topograph.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(Topograph, m) {
    m.doc() = "Highly Efficient C++ Topograph implementation"; // optional module docstring

    //m.def("compute_batch_loss", &compute_batch_loss, "Computes the loss for a batch of prediction and ground truth pair", py::return_value_policy::take_ownership);
    m.def("compute_single_loss", &compute_single_loss, py::arg("argmax_pred").noconvert(), py::arg("argmax_gt").noconvert(),py::arg("num_classes"), py::call_guard<py::gil_scoped_release>(), py::return_value_policy::take_ownership);
    m.def("get_relabel_indices", &get_relabel_indices, py::arg("labelled_comps").noconvert(), py::arg("critical_nodes").noconvert(), py::arg("cluster_lengths").noconvert(), py::call_guard<py::gil_scoped_release>(), py::return_value_policy::take_ownership);
}