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

#include "cudaqlib/gse.h"
#include "../utils/type_casters.h"
#include "../utils/utils.h"

namespace py = pybind11;

namespace cudaq::gse {

void bindGse(py::module &mod) {

  auto gse = mod.def_submodule("gse");

  gse.def(
      "vqe",
      [](const py::function &kernel, cudaq::spin_op op,
         std::vector<double> initial_parameters, py::kwargs options) {
        optimizer_options optOptions;
        optOptions.shots = cudaq::getValueOr<int>(options, "shots", -1);
        if (options.contains("max_iterations"))
          optOptions.max_iterations = cudaq::getValueOr<int>(
              options, "max_iterations", -1); // Or is just a dummy

        optOptions.verbose =
            cudaq::getValueOr<bool>(options, "verbose", false);
        auto optimizerName =
            cudaq::getValueOr<std::string>(options, "optimizer", "cobyla");
        auto optimizer = cudaq::optim::optimizer::get(optimizerName);
        auto kernelWrapper = [&](std::vector<double> x) { kernel(x); };

        if (!optimizer->requiresGradients()) {
          auto result = cudaq::gse::vqe(kernelWrapper, op, *optimizer.get(),
                                        initial_parameters, optOptions);
          return py::make_tuple(result.energy, result.optimal_parameters,
                                result.iteration_data);
        }

        auto gradientName = cudaq::getValueOr<std::string>(options, "gradient",
                                                            "parameter_shift");
        auto gradient = cudaq::optim::observe_gradient::get(gradientName);
        gradient->set_parameterized_kernel(kernelWrapper);
        gradient->set_spin_op(op);

        auto result =
            cudaq::gse::vqe(kernelWrapper, op, *optimizer.get(),
                            *gradient.get(), initial_parameters, optOptions);
        return py::make_tuple(result.energy, result.optimal_parameters,
                              result.iteration_data);
      },
      py::arg("kernel"), py::arg("spin_op"), py::arg("initial_parameters"), "");
}

} // namespace cudaq::gse