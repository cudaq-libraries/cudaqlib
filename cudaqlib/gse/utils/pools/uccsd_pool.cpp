/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "uccsd_pool.h"
#include "cudaqlib/kernels/uccsd.h"
namespace cudaq {

std::vector<spin_op>
uccsd::generate(const std::unordered_map<std::string, std::any> &config) const {
  auto iter = findIter({"num-qubits", "num_qubits"},
                       config); // config.find("num-qubits");
  if (!iter.has_value())
    throw std::runtime_error("uccsd_pool requires num-qubits config "
                             "parameter of type std::size_t.");
  auto numQubits = getIntLike(iter.value()->second);

  iter = findIter({"num-electrons", "num_electrons"}, config);
  if (!iter.has_value())
    throw std::runtime_error("uccsd_pool requires num-electrons config "
                             "parameter of type std::size_t.");
  auto numElectrons = getIntLike(iter.value()->second);

  auto [singlesAlpha, singlesBeta, doublesMixed, doublesAlpha, doublesBeta] =
      cudaq::get_uccsd_excitations(numElectrons, numQubits);

  std::vector<spin_op> ops;
  using namespace cudaq::spin;

  auto addSinglesExcitation = [numQubits](std::vector<spin_op> &ops,
                                          std::size_t p, std::size_t q) {
    spin_op o(numQubits);
    ops.emplace_back(o * spin::y(p) * spin::x(q));
    ops.emplace_back(o * spin::x(p) * spin::y(q));
  };

  auto addDoublesExcitation = [numQubits](std::vector<spin_op> &ops,
                                          std::size_t p, std::size_t q,
                                          std::size_t r, std::size_t s) {
    spin_op o(numQubits);
    ops.emplace_back(o * spin::x(p) * spin::x(q) * spin::x(r) * spin::y(s));
    ops.emplace_back(o * spin::x(p) * spin::x(q) * spin::y(r) * spin::x(s));
    ops.emplace_back(o * spin::x(p) * spin::y(q) * spin::y(r) * spin::y(s));
    ops.emplace_back(o * spin::x(p) * spin::y(q) * spin::x(r) * spin::x(s));

    ops.emplace_back(o * spin::y(p) * spin::x(q) * spin::x(r) * spin::x(s));
    ops.emplace_back(o * spin::y(p) * spin::x(q) * spin::y(r) * spin::y(s));
    ops.emplace_back(o * spin::y(p) * spin::y(q) * spin::x(r) * spin::y(s));
    ops.emplace_back(o * spin::y(p) * spin::y(q) * spin::y(r) * spin::x(s));
  };

  for (auto &sa : singlesAlpha)
    addSinglesExcitation(ops, sa[0], sa[1]);
  for (auto &sa : singlesBeta)
    addSinglesExcitation(ops, sa[0], sa[1]);

  for (auto &d : doublesMixed)
    addDoublesExcitation(ops, d[0], d[1], d[2], d[3]);
  for (auto &d : doublesAlpha)
    addDoublesExcitation(ops, d[0], d[1], d[2], d[3]);
  for (auto &d : doublesBeta)
    addDoublesExcitation(ops, d[0], d[1], d[2], d[3]);

  return ops;
}

} // namespace cudaq