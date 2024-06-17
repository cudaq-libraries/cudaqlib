/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "fermion_op.h"
#include <cassert>

#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>

namespace cudaq::operators {
one_body_integrals::one_body_integrals(const std::vector<std::size_t> &shape)
    : shape(shape) {
  assert(shape.size() == 2);
  ownedData = std::unique_ptr<std::complex<double>>(
      new std::complex<double>[shape[0] * shape[1]]);
}

void one_body_integrals::set_data(
    const std::vector<std::complex<double>> &data) {
  std::complex<double> *ptr = nullptr;
  if (ownedData)
    ptr = ownedData.get();
  else
    ptr = unOwnedData;
  std::copy(data.begin(), data.end(), ptr);
}

std::complex<double> &one_body_integrals::operator()(std::size_t p,
                                                     std::size_t q) const {
  auto *local = unOwnedData == nullptr ? ownedData.get() : unOwnedData;
  return xt::adapt(local, shape[0] * shape[1], xt::no_ownership(), shape)(p, q);
}

void one_body_integrals::dump() {
  auto *local = unOwnedData == nullptr ? ownedData.get() : unOwnedData;
  std::cerr << xt::adapt(local, shape[0] * shape[1], xt::no_ownership(), shape)
            << '\n';
}

two_body_integrals::two_body_integrals(const std::vector<std::size_t> &shape)
    : shape(shape) {
  assert(shape.size() == 4);
  ownedData = std::unique_ptr<std::complex<double>>(
      new std::complex<double>[shape[0] * shape[1] * shape[2] * shape[3]]);
}

void two_body_integrals::set_data(
    const std::vector<std::complex<double>> &data) {
  std::complex<double> *ptr = nullptr;
  if (ownedData)
    ptr = ownedData.get();
  else
    ptr = unOwnedData;
  std::copy(data.begin(), data.end(), ptr);
}

std::complex<double> &two_body_integrals::operator()(std::size_t p,
                                                     std::size_t q,
                                                     std::size_t r,
                                                     std::size_t s) const {
  auto *local = unOwnedData == nullptr ? ownedData.get() : unOwnedData;
  return xt::adapt(local, shape[0] * shape[1] * shape[2] * shape[3],
                   xt::no_ownership(), shape)(p, q, r, s);
}

void two_body_integrals::dump() {
  auto *local = unOwnedData == nullptr ? ownedData.get() : unOwnedData;
  std::cerr << xt::adapt(local, shape[0] * shape[1] * shape[2] * shape[3],
                         xt::no_ownership(), shape)
            << '\n';
}
} // namespace cudaq::operators