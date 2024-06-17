/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <limits>
#include <pybind11/complex.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "cudaqlib/optim.h"
#include "../utils/type_casters.h"
#include "../utils/utils.h"

namespace py = pybind11;

namespace cudaq::optim {

void bindOptim(py::module &mod) {

  auto optim = mod.def_submodule("optim");
  py::enum_<observe_execution_type>(optim, "ObserveExecutionType")
      .value("function", observe_execution_type::function)
      .value("gradient", observe_execution_type::gradient);

  py::class_<observe_iteration>(optim, "ObserveIteration")
      .def_readonly("parameters", &observe_iteration::parameters)
      .def_readonly("result", &observe_iteration::result)
      .def_readonly("type", &observe_iteration::type);

  optim.def(
      "optimize",
      [](const py::function &function, std::vector<double> xInit,
         std::string method, py::kwargs options) {
        optimizer_options optOptions;
        optOptions.initial_parameters = xInit;

        if (!cudaq::optim::optimizer::is_registered(method))
          throw std::runtime_error(
              method + " is not a valid, registered cudaq-x optimizer.");

        auto opt = cudaq::optim::optimizer::get(method);
        auto result = opt->optimize(
            xInit.size(), optOptions,
            [&](std::vector<double> x, std::vector<double> &grad) {
              // Call the function.
              auto ret = function(x);
              // Does it return a tuple?
              auto isTupleReturn = py::isinstance<py::tuple>(ret);
              // If we don't need gradients, and it does, just grab the value
              // and return.
              if (!opt->requiresGradients() && isTupleReturn)
                return ret.cast<py::tuple>()[0].cast<double>();
              // If we dont need gradients and it doesn't return tuple, then
              // just pass what we got.
              if (!opt->requiresGradients() && !isTupleReturn)
                return ret.cast<double>();

              // Throw an error if we need gradients and they weren't provided.
              if (opt->requiresGradients() && !isTupleReturn)
                throw std::runtime_error(
                    "Invalid return type on objective function, must return "
                    "(float,list[float]) for gradient-based optimizers");

              // If here, we require gradients, and the signature is right.
              auto tuple = ret.cast<py::tuple>();
              auto val = tuple[0];
              auto gradIn = tuple[1].cast<py::list>();
              for (std::size_t i = 0; i < gradIn.size(); i++)
                grad[i] = gradIn[i].cast<double>();

              return val.cast<double>();
            });

        return result;
      },
      py::arg("function"), py::arg("initial_parameters"),
      py::arg("method") = "cobyla", "");
}

} // namespace cudaq::optim
