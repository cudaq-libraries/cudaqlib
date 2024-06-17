/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/spin_op.h"

namespace cudaq::gse {
class operator_pool {
private:
  std::vector<spin_op> pool;

public:
  operator_pool() = default;
  operator_pool(const std::vector<spin_op> &opPool) : pool(opPool) {}
  virtual std::vector<spin_op> generate() const { return pool; }
};

namespace fermion {
inline spin_op adag(std::size_t numQubits, std::size_t j) {
  spin_op zprod(numQubits);
  for (std::size_t k = 0; k < j; k++)
    zprod *= spin::z(k);
  return 0.5 * zprod * (spin::x(j) - std::complex<double>{0, 1} * spin::y(j));
}

inline spin_op a(std::size_t numQubits, std::size_t j) {
  spin_op zprod(numQubits);
  for (std::size_t k = 0; k < j; k++)
    zprod *= spin::z(k);
  return 0.5 * zprod * (spin::x(j) + std::complex<double>{0, 1} * spin::y(j));
}
} // namespace fermion

} // namespace cudaq::gse
