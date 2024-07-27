/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq.h"
#include "cudaq/qis/qubit_qis.h"
#include "cudaqlib/gse/utils/operator_pool.h"
#include "cudaqlib/gse/vqe/vqe.h"

#include <functional>

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

inline __qpu__ void gradientKernel(cudaq::state &initState) {
  cudaq::qvector q{initState};
}

template <typename InitialState>
auto adapt_vqe(const InitialState &initialState, const spin_op &H,
               const std::vector<spin_op> &poolList,
               optim::optimizer &optimizer, observe_gradient *gradient,
               const adapt_options options = adapt_options()) {

  AdaptKernel kernel;
  std::vector<spin_op> trotterList;
  std::vector<double> thetas;
  auto numQubits = H.num_qubits();
  // Assumes each rank can see numQpus, models a distributed
  // architecture where each rank is a compute node, and each node
  // has numQpus GPUs available. Each GPU is indexed 0, 1, 2, ..
  std::size_t numQpus = cudaq::get_platform().num_qpus();
  std::size_t numRanks =
      cudaq::mpi::is_initialized() ? cudaq::mpi::num_ranks() : 1;
  std::size_t rank = cudaq::mpi::is_initialized() ? cudaq::mpi::rank() : 0;
  double energy = 0.0, lastNorm = std::numeric_limits<double>::max();

  // poolList is split into numRanks chunks, and each chunk can be
  // further parallelized across numQpus.
  // Compute the [H,Oi]
  std::vector<spin_op> commutators;
  std::size_t total_elements = poolList.size();
  std::size_t elements_per_rank = total_elements / numRanks;
  std::size_t remainder = total_elements % numRanks;

  std::size_t start = rank * elements_per_rank + std::min(rank, remainder);
  std::size_t end = start + elements_per_rank + (rank < remainder ? 1 : 0);

  for (int i = start; i < end; i++) {
    auto op = poolList[i];
    commutators.emplace_back(H * op - op * H);
  }

  // Start of with the initial |psi_n>
  cudaq::state state =
      get_state(kernel, numQubits, initialState, thetas, trotterList);

  while (true) {

    // Step 1 - compute <psi|[H,Oi]|psi> vector
    std::vector<double> gradients;
    double gradNorm = 0.0;
    std::vector<async_observe_result> resultHandles;
    for (std::size_t i = 0, qpuCounter = 0; i < commutators.size(); i++) {
      if (qpuCounter % numQpus == 0)
        qpuCounter = 0;

      resultHandles.emplace_back(
          observe_async(qpuCounter++, gradientKernel, commutators[i], state));
    }

    std::vector<observe_result> results;
    for (auto &handle : resultHandles)
      results.emplace_back(handle.get());

    // Get the gradient results
    std::transform(results.begin(), results.end(),
                   std::back_inserter(gradients),
                   [](auto &&el) { return std::fabs(el.expectation()); });

    // Compute the local gradient norm
    double norm = 0.0;
    for (auto &g : gradients)
      norm += g * g;

    // All ranks have a norm, need to reduce that across all
    if (mpi::is_initialized())
      norm = cudaq::mpi::all_reduce(norm, std::plus<double>());

    // All ranks have a max gradient and index
    auto iter = std::max_element(gradients.begin(), gradients.end());
    double maxGrad = *iter;
    auto maxOpIdx = std::distance(gradients.begin(), iter);
    if (mpi::is_initialized()) {
      std::vector<int> allMaxOpIndices(numRanks);
      std::vector<double> allMaxGrads(numRanks);
      cudaq::mpi::all_gather(allMaxGrads, {*iter});
      cudaq::mpi::all_gather(allMaxOpIndices, {static_cast<int>(maxOpIdx)});
      std::size_t tmp = 0;
      double tmpGrad = 0.0;
      for (std::size_t i = 0; i < allMaxGrads.size(); i++) {
        if (allMaxGrads[i] > tmpGrad) {
          tmpGrad = allMaxGrads[i];
          tmp = allMaxOpIndices[i];
        }
      }

      maxOpIdx = tmp;
    }

    // Convergence is reached if gradient values are small
    if (std::sqrt(std::fabs(norm)) < options.grad_norm_tolerance ||
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
    state = get_state(wrapperKernel, thetas);
  }

  return energy;
}

template <typename InitialState>
auto adapt_vqe(const InitialState &initialState, const spin_op &H,
               const std::vector<spin_op> &operatorPool,
               optim::optimizer &optimizer,
               const adapt_options options = adapt_options()) {
  return adapt_vqe(initialState, H, operatorPool, optimizer, nullptr, options);
}

template <typename InitialState>
auto adapt_vqe(const InitialState &initialState, const spin_op &H,
               const std::vector<spin_op> &operatorPool,
               const adapt_options options = adapt_options()) {
  cudaq::optim::cobyla optimizer;
  return adapt_vqe(initialState, H, operatorPool, optimizer, options);
}

} // namespace cudaq::gse