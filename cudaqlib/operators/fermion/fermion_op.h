/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <complex>
#include <memory>
#include <vector>

namespace cudaq::operators {

/// @brief The `one_body_integrals` provide simple holder type
/// for the `h_pq` coefficients for the second quantized molecular Hamiltonian.
class one_body_integrals {
private:
  std::unique_ptr<std::complex<double>> ownedData;
  std::complex<double> *unOwnedData = nullptr;

public:
  std::vector<std::size_t> shape;
  one_body_integrals(std::complex<double> *externalData,
                     const std::vector<std::size_t> &shape)
      : unOwnedData(externalData), shape(shape) {}
  one_body_integrals(const std::vector<std::size_t> &shape);
  std::complex<double> &operator()(std::size_t i, std::size_t j) const;
  void dump();
  void set_data(const std::vector<std::complex<double>> &data);
  std::complex<double> *raw_data() {
    if (ownedData)
      return ownedData.get();
    return unOwnedData;
  }
};

/// @brief The `two_body_integrals` provide simple holder type
/// for the `h_pqrs` coefficients for the second quantized molecular
/// Hamiltonian.
class two_body_integrals {
private:
  std::unique_ptr<std::complex<double>> ownedData;
  std::complex<double> *unOwnedData = nullptr;

public:
  std::vector<std::size_t> shape;
  two_body_integrals(std::complex<double> *externalData,
                     const std::vector<std::size_t> &shape)
      : unOwnedData(externalData), shape(shape) {}
  two_body_integrals(const std::vector<std::size_t> &shape);
  std::complex<double> &operator()(std::size_t p, std::size_t q, std::size_t r,
                                   std::size_t s) const;
  void dump();
  void set_data(const std::vector<std::complex<double>> &data);
  std::complex<double> *raw_data() {
    if (ownedData)
      return ownedData.get();
    return unOwnedData;
  }
};

/// @brief The `fermion_op` is a simple wrapper around one
/// and two body integral tensors describing a second quantized
/// fermionic Hamiltonian.
class fermion_op {
public:
  /// @brief The one-body integrals, `h[p,q]`.
  one_body_integrals hpq;

  /// @brief The two-body integrals, `h[p,q,r,s]`.
  two_body_integrals hpqrs;

  /// @brief The constant energy term.
  double constant;

  fermion_op(std::size_t numQubits, double c)
      : hpq(one_body_integrals({numQubits, numQubits})),
        hpqrs(two_body_integrals({numQubits, numQubits, numQubits, numQubits})),
        constant(c) {}
};

/// @brief The `fermion_pe_op` is a simple wrapper around one
/// body integral tensors describing a second quantized
/// fermionic of the polarizable embedding Hamiltonian.

class fermion_pe_op{
  public:
  /// @brief The one-body integrals, `h[p,q]`.
  one_body_integrals vpq;

  fermion_pe_op(std::size_t numQubits)
      : vpq(one_body_integrals({numQubits, numQubits})){}

};

} // namespace cudaq::operators