/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaqlib/gse/vqe/vqe.h"
#include "cudaq/qis/qubit_qis.h"

#include "operator_pool.h"

namespace cudaq::gse {

struct adapt_options {
  double grad_norm_tolerance = 1e-5;
  double grad_norm_magnitude = 1e-1;
  bool verbose = false;
};

struct AdaptKernel {
  void operator()(std::size_t numQubits, cudaq::takes_qvector auto &&statePrep,
                  const std::vector<double> &thetas,
                  const std::vector<spin_op> &trotterOpList) __qpu__ {
    cudaq::qvector q(numQubits);
    statePrep(q);
    for (std::size_t i = 0; i < thetas.size(); i++)
      for (auto &term : trotterOpList[i])
        exp_pauli(thetas[i] * term.get_coefficient().imag(), q,
                  term.to_string(false).c_str());
  }
};

template <typename InitialState>
auto adapt_vqe(const InitialState &initialState, const spin_op &H,
               const operator_pool &operatorPool, optim::optimizer &optimizer,
               observe_gradient *gradient,
               const adapt_options options = adapt_options()) {

  AdaptKernel kernel;
  std::vector<spin_op> trotterList;
  std::vector<double> thetas;
  auto numQubits = H.num_qubits();
  auto poolList = operatorPool.generate();
  double energy = 0.0, lastNorm = std::numeric_limits<double>::max();

  // Compute the [H,Oi]
  std::vector<spin_op> commutators;
  for (auto &op : poolList)
    commutators.emplace_back(H * op - op * H);

  while (true) {

    // Step 1 - compute <psi|[H,Oi]|psi> vector
    std::vector<double> gradients;
    double gradNorm = 0.0;
    std::vector<observe_result> results;
    for (std::size_t i = 0; i < commutators.size(); i++)
      results.emplace_back(observe(kernel, commutators[i], numQubits,
                                   initialState, thetas, trotterList));

    // Get the gradient results
    std::transform(results.begin(), results.end(),
                   std::back_inserter(gradients),
                   [](auto &&el) { return std::fabs(el.expectation()); });

    // Compute the local gradient norm
    double norm = 0.0;
    for (auto &g : gradients)
      norm += g * g;

    auto iter = std::max_element(gradients.begin(), gradients.end());
    auto maxOpIdx = std::distance(gradients.begin(), iter);

    // Convergence is reached if gradient values are small
    if (std::sqrt(std::fabs(norm)) < options.grad_norm_magnitude ||
        std::fabs(lastNorm - norm) < options.grad_norm_tolerance)
      break;

    // Set the norm for the next iteration's check
    lastNorm = norm;

    // Use the operator from the pool
    trotterList.push_back(poolList[maxOpIdx]);
    thetas.push_back(0.0);

    // All VQE kernels have to have the std::vector<double> signature,
    // so wrap AdaptKernel here
    auto wrapperKernel = [&](std::vector<double> thetas) __qpu__ {
      kernel(numQubits, initialState, thetas, trotterList);
    };

    // Run gradient-based, or non-gradient based VQE
    vqe_result result;
    if (gradient != nullptr)
      result = cudaq::gse::vqe(wrapperKernel, H, optimizer, *gradient, thetas);
    else if (optimizer.requiresGradients()) {
      cudaq::optim::parameter_shift defaultGradient(wrapperKernel, H);
      result =
          cudaq::gse::vqe(wrapperKernel, H, optimizer, defaultGradient, thetas);
    } else
      result = cudaq::gse::vqe(wrapperKernel, H, optimizer, thetas);

    // Set the new optimzal parameters
    thetas = result.optimal_parameters;
    energy = result.energy;
  }

  return energy;
}

template <typename InitialState>
auto adapt_vqe(const InitialState &initialState, const spin_op &H,
               const operator_pool &operatorPool, optim::optimizer &optimizer,
               const adapt_options options = adapt_options()) {
  return adapt_vqe(initialState, H, operatorPool, optimizer, nullptr, options);
}

template <typename InitialState>
auto adapt_vqe(const InitialState &initialState, const spin_op &H,
               const operator_pool &operatorPool,
               const adapt_options options = adapt_options()) {
  cudaq::optim::cobyla optimizer;
  return adapt_vqe(initialState, H, operatorPool, optimizer, options);
}

} // namespace cudaq::gse