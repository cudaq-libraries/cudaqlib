/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cmath>
#include <gtest/gtest.h>

#include "cudaqlib/gse.h"
#include "cudaq.h"

TEST(GSETester, checkSimpleAdapt) {

  std::vector<double> h2_data{
      3, 1, 1, 3, 0.0454063,  0,  2, 0, 0, 0, 0.17028,    0,
      0, 0, 2, 0, -0.220041,  -0, 1, 3, 3, 1, 0.0454063,  0,
      0, 0, 0, 0, -0.106477,  0,  0, 2, 0, 0, 0.17028,    0,
      0, 0, 0, 2, -0.220041,  -0, 3, 3, 1, 1, -0.0454063, -0,
      2, 2, 0, 0, 0.168336,   0,  2, 0, 2, 0, 0.1202,     0,
      0, 2, 0, 2, 0.1202,     0,  2, 0, 0, 2, 0.165607,   0,
      0, 2, 2, 0, 0.165607,   0,  0, 0, 2, 2, 0.174073,   0,
      1, 1, 3, 3, -0.0454063, -0, 15};
  cudaq::spin_op h(h2_data, 4);

  cudaq::gse::spin_complement_gsd pool(h.num_qubits() / 2);

  auto initialState = [&](cudaq::qvector<> &q) __qpu__ {
    for (std::size_t i = 0; i < 2; i++)
      x(q[i]);
  };

  auto energy = cudaq::gse::adapt_vqe(initialState, h, pool,
                                      {.grad_norm_tolerance = 1e-4});
  EXPECT_NEAR(energy, -1.13, 1e-2);
}