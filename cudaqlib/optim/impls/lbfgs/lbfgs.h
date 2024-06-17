/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaqlib/optim/optimizer.h"

namespace cudaq::optim {

/// @brief The limited-memory Broyden-Fletcher-Goldfarb-Shanno
/// gradient based black-box function optimizer.
class lbfgs : public details::optimizer_impl<lbfgs> {
public:
  using optimizer::optimize;

  /// @brief Return true indicating this optimizer requires an
  /// optimization functor that produces gradients.
  bool requiresGradients() override { return true; }

  /// @brief Optimize the provided function according to the
  /// LBFGS algorithm.
  optimization_result
  optimize(std::size_t dim, const optimizer_options &options,
           const optimizable_function &opt_function) override;

  CUDAQ_REGISTER_OPTIMIZER(lbfgs)
};
} // namespace cudaq::optim