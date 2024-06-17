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

/// @brief The
class cobyla : public details::optimizer_impl<cobyla> {
public:
  using optimizer::optimize;

  /// @brief Return true indicating this optimizer requires an
  /// optimization functor that produces gradients.
  bool requiresGradients() override { return false; }

  /// @brief Optimize the provided function according to the
  /// cobyla algorithm.
  optimization_result
  optimize(std::size_t dim, const optimizer_options &options,
           const optimizable_function &opt_function) override;

  CUDAQ_REGISTER_OPTIMIZER(cobyla)
};
} // namespace cudaq::optim