/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaqlib/optim/observe_gradient.h"

namespace cudaq::optim {

class central_difference
    : public details::observe_gradient_impl<central_difference> {
protected:
  std::size_t
  getRequiredNumExpectationComputations(const std::vector<double> &x) override {
    return 2 * x.size();
  }

public:
  double step = 1e-4;

  void calculateGradient(const std::vector<double> &x, std::vector<double> &dx,
                         double exp_h) override {
    auto tmpX = x;
    for (std::size_t i = 0; i < x.size(); i++) {
      // increase value to x_i + dx_i
      tmpX[i] += step;
      auto px = expectation(tmpX);
      // decrease the value to x_i - dx_i
      tmpX[i] -= 2 * step;
      auto mx = expectation(tmpX);
      dx[i] = (px - mx) / (2. * step);
    }
  }

  CUDAQ_REGISTER_GRADIENT(central_difference)
};
} // namespace cudaq::optim