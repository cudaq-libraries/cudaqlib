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

class parameter_shift : public details::observe_gradient_impl<parameter_shift> {
protected:
  std::size_t
  getRequiredNumExpectationComputations(const std::vector<double> &x) override {
    return 2 * x.size();
  }

public:
  double shiftScalar = 0.5;

  void calculateGradient(const std::vector<double> &x, std::vector<double> &dx,
                         double exp_h) override {
    auto tmpX = x;
    for (std::size_t i = 0; i < x.size(); i++) {
      // increase value to x_i + (shiftScalar * pi)
      tmpX[i] += shiftScalar * M_PI;
      auto px = expectation(tmpX);
      // decrease value to x_i - (shiftScalar * pi)
      tmpX[i] -= 2 * shiftScalar * M_PI;
      auto mx = expectation(tmpX);
      dx[i] = (px - mx) / 2.;
    }
  }

  CUDAQ_REGISTER_GRADIENT(parameter_shift)
};
} // namespace cudaq::optim