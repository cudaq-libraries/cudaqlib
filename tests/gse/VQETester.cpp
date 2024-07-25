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
#include "cudaqlib/gse.h"

TEST(GSETester, checkVqe) {

  using namespace cudaq::spin;

  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);
  auto ansatz = [](std::vector<double> theta) __qpu__ {
    cudaq::qvector q(2);
    x(q[0]);
    ry(theta[0], q[1]);
    x<cudaq::ctrl>(q[1], q[0]);
  };

  {
    cudaq::optim::cobyla optimizer;
    auto result = cudaq::gse::vqe(ansatz, h, optimizer, {0.0});
    EXPECT_NEAR(result.energy, -1.748, 1e-3);
  }

  {
    cudaq::optim::lbfgs optimizer;
    cudaq::optim::central_difference gradient(ansatz, h);
    auto [energy, params, data] =
        cudaq::gse::vqe(ansatz, h, optimizer, gradient, {0.0});
    EXPECT_NEAR(energy, -1.748, 1e-3);
  }

  {
    // Test how one can handle non-standard kernel signature
    cudaq::optim::lbfgs optimizer;
    cudaq::optim::central_difference gradient(ansatz, h);
    auto ansatzNonStdSignature = [](double theta, int N) __qpu__ {
      cudaq::qvector q(N);
      x(q[0]);
      ry(theta, q[1]);
      x<cudaq::ctrl>(q[1], q[0]);
    };

    constexpr int N = 2;

    // Wrap the kernel in another kernel with the standard signature
    auto result = cudaq::gse::vqe(
        [&](std::vector<double> x) __qpu__ { ansatzNonStdSignature(x[0], N); },
        h, optimizer, gradient, {0.0});

    EXPECT_NEAR(result.energy, -1.748, 1e-3);
    EXPECT_TRUE(result.iteration_data.size() > 1);
  }

  // Handle shots-based simulation
  {
    cudaq::set_random_seed(13);
    cudaq::optim::cobyla optimizer;
    auto result = cudaq::gse::vqe(ansatz, h, optimizer,
                                  std::vector<double>{0.0}, {.shots = 10000});
    printf("TEST %lf\n", result.energy);
    result.iteration_data[0].result.dump();
    EXPECT_TRUE(result.energy > -2.0 && result.energy < -1.5);
  }

  // Handle shots-based simulation with gradient
  {
    cudaq::set_random_seed(13);
    cudaq::optim::lbfgs optimizer;
    cudaq::optim::parameter_shift gradient(ansatz, h);
    auto result = cudaq::gse::vqe(ansatz, h, optimizer, gradient, {0.0},
                                  {.shots = 10000});
    printf("TEST %lf\n", result.energy);
    result.iteration_data[0].result.dump();
    for (auto &o : result.iteration_data) {
      printf("Type: %s\n", static_cast<int>(o.type) ? "gradient" : "function");
      o.result.dump();
    }
    EXPECT_TRUE(result.energy > -2.0 && result.energy < -1.5);
  }
}

// void so4(cudaq::qubit &q, cudaq::qubit &r, const std::vector<double> &thetas)
// {
//   ry(thetas[0], q);
//   ry(thetas[1], r);

//   h(r);
//   x<cudaq::ctrl>(q, r);
//   h(r);

//   ry(thetas[2], q);
//   ry(thetas[3], r);

//   h(r);
//   x<cudaq::ctrl>(q, r);
//   h(r);

//   ry(thetas[4], q);
//   ry(thetas[5], r);

//   h(r);
//   x<cudaq::ctrl>(q, r);
//   h(r);
// }

// TEST(VQETester, checkH2) {

//   std::vector<double> h2_data{
//       3, 1, 1, 3, 0.0454063,  0,  2, 0, 0, 0, 0.17028,    0,
//       0, 0, 2, 0, -0.220041,  -0, 1, 3, 3, 1, 0.0454063,  0,
//       0, 0, 0, 0, -0.106477,  0,  0, 2, 0, 0, 0.17028,    0,
//       0, 0, 0, 2, -0.220041,  -0, 3, 3, 1, 1, -0.0454063, -0,
//       2, 2, 0, 0, 0.168336,   0,  2, 0, 2, 0, 0.1202,     0,
//       0, 2, 0, 2, 0.1202,     0,  2, 0, 0, 2, 0.165607,   0,
//       0, 2, 2, 0, 0.165607,   0,  0, 0, 2, 2, 0.174073,   0,
//       1, 1, 3, 3, -0.0454063, -0, 15};
//   cudaq::spin_op h(h2_data, 4);

//   struct so4_fabric {
//     void operator()(std::vector<double> params) __qpu__ {
//       cudaq::qreg q(4);

//       x(q[0]);
//       x(q[1]);

//       const int block_size = 2;
//       int counter = 0;
//       for (int i = 0; i < 2; i++) {
//         // first layer of so4 blocks (even)
//         for (int k = 0; k < 4; k += 2) {
//           auto subq = q.slice(k, block_size);
//           auto so4_params = cudaq::slice_vector(params, counter, 6);
//           so4(subq[0], subq[1], so4_params);
//           counter += 6;
//         }

//         // second layer of so4 blocks (odd)
//         for (int k = 1; k + block_size < 4; k += 2) {
//           auto subq = q.slice(k, block_size);
//           auto so4_params = cudaq::slice_vector(params, counter, 6);
//           so4(subq[0], subq[1], so4_params);
//           counter += 6;
//         }
//       }
//     }
//   };

//   int n_layers = 2, n_qubits = h.num_qubits(), block_size = 2, p_counter = 0;
//   int n_blocks_per_layer = 2 * (n_qubits / block_size) - 1;
//   int n_params = n_layers * 6 * n_blocks_per_layer;

//   so4_fabric ansatz;
//   cudaq::optim::lbfgs optimizer;
//   cudaq::optim::central_difference gradient(ansatz, h);
//   optimizer_options options;
//   options.max_iterations = 20;
//   options.initial_parameters = cudaq::random_vector(-M_PI, M_PI, n_params);
//   options.verbose = true;
//   auto result =
//       cudaq::gse::vqe(ansatz, h, optimizer, gradient, n_params, options);
//   printf("result %lf\n", result.energy);
//   EXPECT_NEAR(result.energy, -1.13, 1e-2);
// }
