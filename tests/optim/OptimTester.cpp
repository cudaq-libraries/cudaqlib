/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cmath>
#include <gtest/gtest.h>

#include "cudaq.h"
#include "cudaqlib/optim.h"

TEST(OptimTester, checkLBFGS) {

  cudaq::optim::lbfgs optimizer;
  // optimizer.verbose = true;
  auto f = [](const std::vector<double> &x, std::vector<double> &dx) {
    dx[0] = -2 * (1 - x[0]) + 400 * (std::pow(x[0], 3) - x[1] * x[0]);
    dx[1] = 200 * (x[1] - std::pow(x[0], 2));
    return 100 * std::pow(x[1] - std::pow(x[0], 2), 2) + std::pow(1 - x[0], 2);
  };

  {
    auto [opt, params] = optimizer.optimize(2, f);
    EXPECT_NEAR(0.0, opt, 1e-3);
    EXPECT_EQ(2, params.size());
    EXPECT_NEAR(1.0, params[0], 1e-3);
    EXPECT_NEAR(1.0, params[1], 1e-3);
  }

  {
    auto nIters = optimizer.history.size();
    // Try to set the tolerance
    auto [opt, params] = optimizer.optimize(2, {.function_tolerance = 1e-3}, f);

    EXPECT_TRUE(optimizer.history.size() < nIters);
  }

  {
    // Try to set the tolerance
    auto [opt, params] = optimizer.optimize(2, {.max_iterations = 12}, f);
    EXPECT_TRUE(optimizer.history.size() == 12);
  }
}
TEST(OptimTester, checkLBFGSFactory) {
  EXPECT_TRUE(cudaq::optim::optimizer::is_registered("lbfgs"));
  auto optimizer = cudaq::optim::optimizer::get("lbfgs");
  // optimizer.verbose = true;
  auto f = [](const std::vector<double> &x, std::vector<double> &dx) {
    dx[0] = -2 * (1 - x[0]) + 400 * (std::pow(x[0], 3) - x[1] * x[0]);
    dx[1] = 200 * (x[1] - std::pow(x[0], 2));
    return 100 * std::pow(x[1] - std::pow(x[0], 2), 2) + std::pow(1 - x[0], 2);
  };

  {
    auto [opt, params] = optimizer->optimize(2, f);
    EXPECT_NEAR(0.0, opt, 1e-3);
    EXPECT_EQ(2, params.size());
    EXPECT_NEAR(1.0, params[0], 1e-3);
    EXPECT_NEAR(1.0, params[1], 1e-3);
  }
}

TEST(OptimTester, checkCOBYLA) {

  cudaq::optim::cobyla optimizer;
  // optimizer.verbose = true;
  auto f = [](const std::vector<double> &x, std::vector<double> &dx) {
    return 100 * std::pow(x[1] - std::pow(x[0], 2), 2) + std::pow(1 - x[0], 2);
  };

  {
    auto [opt, params] = optimizer.optimize(
        2, {.initial_parameters = std::vector<double>(2, 1.)}, f);
    EXPECT_NEAR(0.0, opt, 1e-3);
    EXPECT_EQ(2, params.size());
    EXPECT_NEAR(1.0, params[0], 1e-3);
    EXPECT_NEAR(1.0, params[1], 1e-3);
  }
}
