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

  iter = findIter({"operatorCoeffs", "operator-coeffs", "operator_coeffs",
                   "coeffs", "thetas"},
                  config);
  std::vector<double> operatorCoeffs;
  if (iter.has_value())
    operatorCoeffs = std::any_cast<std::vector<double>>(iter.value()->second);

  std::size_t spin = 0;
  iter = findIter({"spin"}, config);
  if (iter.has_value())
    spin = getIntLike(iter.value()->second);

  auto [singlesAlpha, singlesBeta, doublesMixed, doublesAlpha, doublesBeta] =
      cudaq::get_uccsd_excitations(numElectrons, spin, numQubits);

  std::vector<spin_op> ops;
  using namespace cudaq::spin;

  auto addSinglesExcitation = [numQubits](std::vector<cudaq::spin_op> &ops,
                                          std::size_t p, std::size_t q) {
    double parity = 1.0;

    cudaq::spin_op o(numQubits);
    for (std::size_t i = p + 1; i < q; i++)
      o *= cudaq::spin::z(i);

    ops.emplace_back(cudaq::spin::y(p) * o * cudaq::spin::x(q));
    ops.emplace_back(cudaq::spin::x(p) * o * cudaq::spin::y(q));
  };

  auto addDoublesExcitation = [numQubits](std::vector<cudaq::spin_op> &ops,
                                          std::size_t p, std::size_t q,
                                          std::size_t r, std::size_t s) {
    cudaq::spin_op parity_a(numQubits), parity_b(numQubits);
    std::size_t i_occ = 0, j_occ = 0, a_virt = 0, b_virt = 0;
    if (p < q && r < s) {
      i_occ = p;
      j_occ = q;
      a_virt = r;
      b_virt = s;
    }

    else if (p > q && r > s) {
      i_occ = q;
      j_occ = p;
      a_virt = s;
      b_virt = r;
    } else if (p < q && r > s) {
      i_occ = p;
      j_occ = q;
      a_virt = s;
      b_virt = r;
    } else if

        (p > q && r < s) {
      i_occ = q;
      j_occ = p;
      a_virt = r;
      b_virt = s;
    }
    for (std::size_t i = i_occ + 1; i < j_occ; i++)
      parity_a *= cudaq::spin::z(i);

    for (std::size_t i = a_virt + 1; i < b_virt; i++)
      parity_b *= cudaq::spin::z(i);

    ops.emplace_back(cudaq::spin::x(i_occ) * parity_a * cudaq::spin::x(j_occ) *
                     cudaq::spin::x(a_virt) * parity_b *
                     cudaq::spin::y(b_virt));
    ops.emplace_back(cudaq::spin::x(i_occ) * parity_a * cudaq::spin::x(j_occ) *
                     cudaq::spin::y(a_virt) * parity_b *
                     cudaq::spin::x(b_virt));
    ops.emplace_back(cudaq::spin::x(i_occ) * parity_a * cudaq::spin::y(j_occ) *
                     cudaq::spin::y(a_virt) * parity_b *
                     cudaq::spin::y(b_virt));
    ops.emplace_back(cudaq::spin::y(i_occ) * parity_a * cudaq::spin::x(j_occ) *
                     cudaq::spin::y(a_virt) * parity_b *
                     cudaq::spin::y(b_virt));
    ops.emplace_back(cudaq::spin::x(i_occ) * parity_a * cudaq::spin::y(j_occ) *
                     cudaq::spin::x(a_virt) * parity_b *
                     cudaq::spin::x(b_virt));
    ops.emplace_back(cudaq::spin::y(i_occ) * parity_a * cudaq::spin::x(j_occ) *
                     cudaq::spin::x(a_virt) * parity_b *
                     cudaq::spin::x(b_virt));
    ops.emplace_back(cudaq::spin::y(i_occ) * parity_a * cudaq::spin::y(j_occ) *
                     cudaq::spin::x(a_virt) * parity_b *
                     cudaq::spin::y(b_virt));
    ops.emplace_back(cudaq::spin::y(i_occ) * parity_a * cudaq::spin::y(j_occ) *
                     cudaq::spin::y(a_virt) * parity_b *
                     cudaq::spin::x(b_virt));
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

  if (operatorCoeffs.empty())
    return ops;

  std::vector<spin_op> retOps;
  for (auto &c : operatorCoeffs) {
    for (auto &op : ops) {
      retOps.push_back(c * op);
    }
  }
  return retOps;
}

} // namespace cudaq