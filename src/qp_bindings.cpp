#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "QP.hpp"

namespace py = pybind11;

PYBIND11_MODULE(qp_module, m) {
    m.doc() = "Python binding for your custom QP solver";

    py::class_<QP>(m, "QP")
        .def(py::init<Eigen::MatrixXd,
                      Eigen::MatrixXd,
                      Eigen::MatrixXd,
                      Eigen::MatrixXd,
                      Eigen::MatrixXd,
                      Eigen::MatrixXd>(),
             py::arg("Q"),
             py::arg("q"),
             py::arg("G"),
             py::arg("h"),
             py::arg("A"),
             py::arg("b"))
        .def("solve", &QP::solve,
             "Run the interior‑point solver")
        .def("get_opt_value", &QP::get_opt_value,
             "Return the objective value ½ xᵀQx + qᵀx")
        .def_property_readonly("x", [](QP &self) { return self.x; },
             "Primal solution vector")
        ;
}