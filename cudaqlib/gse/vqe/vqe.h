/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaqlib/optim.h"

using namespace cudaq::optim;

namespace cudaq::gse {

/// @brief A vqe_result encapsulates all the data produced
/// by a standard variational quantum eigensolver execution. It
/// provides the programmer with the optimal energy and parameters
/// as well as a list of all execution data at each iteration.
struct vqe_result {
  double energy;
  std::vector<double> optimal_parameters;
  std::vector<observe_iteration> iteration_data;
  operator double() { return energy; }

  // FIXME add to/from file functionality
};

/// @brief Compute the minimal eigenvalue of the given Hamiltonian with VQE.
/// @details Given a quantum kernel of signature `void(std::vector<double>)`,
/// run the variational quantum eigensolver routine to compute
/// the minimum eigenvalue of the specified hermitian `spin_op`.
template <typename QuantumKernel>
  requires std::invocable<QuantumKernel, std::vector<double>>
vqe_result vqe(QuantumKernel &&kernel, const spin_op &hamiltonian,
               optim::optimizer &optimizer, observe_gradient &gradient,
               const std::vector<double> &initial_parameters,
               optimizer_options options = optimizer_options()) {
  if (!optimizer.requiresGradients())
    throw std::runtime_error("[vqe] provided optimizer does not require "
                             "gradients, yet gradient instance provided.");

  options.initial_parameters = initial_parameters;
  std::vector<observe_iteration> data;

  /// Run the optimization
  auto [groundEnergy, optParams] = optimizer.optimize(
      initial_parameters.size(), options,
      [&](const std::vector<double> &x, std::vector<double> &dx) {
        auto res = cudaq::observe(options.shots, kernel, hamiltonian, x);
        if (options.verbose)
          printf("<H> = %.12lf\n", res.expectation());
        data.emplace_back(x, res, observe_execution_type::function);
        gradient.compute(x, dx, res.expectation(), options.shots);
        for (auto datum : gradient.data)
          data.push_back(datum);

        return res.expectation();
      });

  return {groundEnergy, optParams, data};
}

template <typename QuantumKernel>
  requires std::invocable<QuantumKernel, std::vector<double>>
vqe_result vqe(QuantumKernel &&kernel, const spin_op &hamiltonian,
               const std::string &optName, const std::string &gradName,
               const std::vector<double> &initial_parameters,
               optimizer_options options = optimizer_options()) {

  if (!cudaq::optim::optimizer::is_registered(optName))
    throw std::runtime_error("provided optimizer is not valid.");

  if (!cudaq::optim::observe_gradient::is_registered(gradName))
    throw std::runtime_error("provided optimizer is not valid.");

  auto optimizer = cudaq::optim::optimizer::get(optName);
  auto gradient = cudaq::optim::observe_gradient::get(gradName);
  gradient->set_parameterized_kernel(kernel);
  gradient->set_spin_op(hamiltonian);

  if (!optimizer->requiresGradients())
    throw std::runtime_error("[vqe] provided optimizer does not require "
                             "gradients, yet gradient instance provided.");

  options.initial_parameters = initial_parameters;
  std::vector<observe_iteration> data;

  /// Run the optimization
  auto [groundEnergy, optParams] = optimizer->optimize(
      initial_parameters.size(), options,
      [&](const std::vector<double> &x, std::vector<double> &dx) {
        auto res = cudaq::observe(options.shots, kernel, hamiltonian, x);
        if (options.verbose)
          printf("<H> = %.12lf\n", res.expectation());
        data.emplace_back(x, res, observe_execution_type::function);
        gradient->compute(x, dx, res.expectation(), options.shots);
        for (auto datum : gradient->data)
          data.push_back(datum);

        return res.expectation();
      });

  return {groundEnergy, optParams, data};
}

template <typename QuantumKernel>
  requires std::invocable<QuantumKernel, std::vector<double>>
vqe_result vqe(QuantumKernel &&kernel, const spin_op &hamiltonian,
               const std::string &optName,
               const std::vector<double> &initial_parameters,
               optimizer_options options = optimizer_options()) {

  if (!cudaq::optim::optimizer::is_registered(optName))
    throw std::runtime_error("provided optimizer is not valid.");

  auto optimizer = cudaq::optim::optimizer::get(optName);

  if (optimizer->requiresGradients())
    throw std::runtime_error("[vqe] provided optimizer requires "
                             "gradients, yet no gradient instance provided.");

  options.initial_parameters = initial_parameters;
  std::vector<observe_iteration> data;

  /// Run the optimization
  auto [groundEnergy, optParams] = optimizer->optimize(
      initial_parameters.size(), options,
      [&](const std::vector<double> &x, std::vector<double> &dx) {
        auto res = cudaq::observe(options.shots, kernel, hamiltonian, x);
        if (options.verbose)
          printf("<H> = %.12lf\n", res.expectation());
        data.emplace_back(x, res, observe_execution_type::function);
        return res.expectation();
      });

  return {groundEnergy, optParams, data};
}

template <typename QuantumKernel>
  requires std::invocable<QuantumKernel, std::vector<double>>
vqe_result vqe(QuantumKernel &&kernel, const spin_op &hamiltonian,
               const std::string &optName, observe_gradient &gradient,
               const std::vector<double> &initial_parameters,
               optimizer_options options = optimizer_options()) {

  if (!cudaq::optim::optimizer::is_registered(optName))
    throw std::runtime_error("provided optimizer is not valid.");

  auto optimizer = cudaq::optim::optimizer::get(optName);
  if (!optimizer->requiresGradients())
    throw std::runtime_error("[vqe] provided optimizer does not require "
                             "gradients, yet gradient instance provided.");

  options.initial_parameters = initial_parameters;
  std::vector<observe_iteration> data;

  /// Run the optimization
  auto [groundEnergy, optParams] = optimizer->optimize(
      initial_parameters.size(), options,
      [&](const std::vector<double> &x, std::vector<double> &dx) {
        auto res = cudaq::observe(options.shots, kernel, hamiltonian, x);
        if (options.verbose)
          printf("<H> = %.12lf\n", res.expectation());
        data.emplace_back(x, res, observe_execution_type::function);
        gradient.compute(x, dx, res.expectation(), options.shots);
        for (auto datum : gradient.data)
          data.push_back(datum);

        return res.expectation();
      });

  return {groundEnergy, optParams, data};
}

template <typename QuantumKernel>
  requires std::invocable<QuantumKernel, std::vector<double>>
vqe_result vqe(QuantumKernel &&kernel, const spin_op &hamiltonian,
               optim::optimizer &optimizer, const std::string &gradName,
               const std::vector<double> &initial_parameters,
               optimizer_options options = optimizer_options()) {

  if (!cudaq::optim::observe_gradient::is_registered(gradName))
    throw std::runtime_error("provided optimizer is not valid.");

  auto gradient = cudaq::optim::observe_gradient::get(gradName);
  gradient->set_parameterized_kernel(kernel);
  gradient->set_spin_op(hamiltonian);

  if (!optimizer.requiresGradients())
    throw std::runtime_error("[vqe] provided optimizer does not require "
                             "gradients, yet gradient instance provided.");

  options.initial_parameters = initial_parameters;
  std::vector<observe_iteration> data;

  /// Run the optimization
  auto [groundEnergy, optParams] = optimizer.optimize(
      initial_parameters.size(), options,
      [&](const std::vector<double> &x, std::vector<double> &dx) {
        auto res = cudaq::observe(options.shots, kernel, hamiltonian, x);
        if (options.verbose)
          printf("<H> = %.12lf\n", res.expectation());
        data.emplace_back(x, res, observe_execution_type::function);
        gradient->compute(x, dx, res.expectation(), options.shots);
        for (auto datum : gradient->data)
          data.push_back(datum);

        return res.expectation();
      });

  return {groundEnergy, optParams, data};
}

template <typename QuantumKernel>
  requires std::invocable<QuantumKernel, std::vector<double>>
vqe_result vqe(QuantumKernel &&kernel, const spin_op &hamiltonian,
               optim::optimizer &optimizer,
               const std::vector<double> &initial_parameters,
               optimizer_options options = optimizer_options()) {

  if (optimizer.requiresGradients())
    throw std::runtime_error("[vqe] provided optimizer does not require "
                             "gradients, yet gradient instance provided.");

  options.initial_parameters = initial_parameters;

  std::vector<observe_iteration> data;

  /// Run the optimization
  auto [groundEnergy, optParams] = optimizer.optimize(
      initial_parameters.size(), options,
      [&](const std::vector<double> &x, std::vector<double> &dx) {
        auto res = cudaq::observe(options.shots, kernel, hamiltonian, x);
        if (options.verbose)
          printf("<H> = %.12lf\n", res.expectation());
        data.emplace_back(x, res, observe_execution_type::function);

        return res.expectation();
      });

  return {groundEnergy, optParams, data};
}

} // namespace cudaq::gse
