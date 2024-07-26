/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "spin_complement_gsd.h"
#include "cudaqlib/operators/fermion/fermion_operators.h"

namespace cudaq {

std::vector<spin_op> spin_complement_gsd::generate(
    const std::unordered_map<std::string, std::any> &config) const {
  auto iter = config.find("num-orbitals");
  if (iter == config.end())
    throw std::runtime_error("spin_complement_gsd requires num-orbitals config "
                             "parameter of type std::size_t.");

  auto numOrbitals = std::any_cast<std::size_t>(iter->second);

  std::vector<spin_op> pool;
  auto numQubits = 2 * numOrbitals;
  std::vector<int> alphaOrbs, betaOrbs;
  for (auto i : range(numOrbitals)) {
    alphaOrbs.push_back(2 * i);
    betaOrbs.push_back(alphaOrbs.back() + 1);
  }

  for (auto p : alphaOrbs) {
    for (auto q : alphaOrbs) {
      if (p >= q)
        continue;
      auto oneElectron =
          operators::adag(numQubits, q) * operators::a(numQubits, p) -
          operators::adag(numQubits, p) * operators::a(numQubits, q);
      oneElectron +=
          operators::adag(numQubits, q + 1) * operators::a(numQubits, p + 1) -
          operators::adag(numQubits, p + 1) * operators::a(numQubits, q + 1);

      std::unordered_map<spin_op::spin_op_term, std::complex<double>> terms;
      oneElectron.for_each_term([&](spin_op &term) {
        auto coeff = term.get_coefficient();
        if (std::fabs(coeff.real()) < 1e-12 && std::fabs(coeff.imag()) < 1e-12)
          return;

        if (std::fabs(coeff.real()) < 1e-12)
          terms.insert({term.get_raw_data().first[0],
                        std::complex<double>{0., coeff.imag()}});
      });

      if (!terms.empty())
        pool.emplace_back(terms);
    }
  }

  int pq = 0;
  for (auto p : alphaOrbs) {
    for (auto q : alphaOrbs) {
      if (p > q)
        continue;

      int rs = 0;
      for (auto r : alphaOrbs) {
        for (auto s : alphaOrbs) {
          if (r > s)
            continue;
          if (pq < rs)
            continue;

          auto twoElectron =
              operators::adag(numQubits, r) * operators::a(numQubits, p) *
                  operators::adag(numQubits, s) * operators::a(numQubits, q) -
              operators::adag(numQubits, q) * operators::a(numQubits, s) *
                  operators::adag(numQubits, p) * operators::a(numQubits, r);
          twoElectron += operators::adag(numQubits, r + 1) *
                             operators::a(numQubits, p + 1) *
                             operators::adag(numQubits, s + 1) *
                             operators::a(numQubits, q + 1) -
                         operators::adag(numQubits, q + 1) *
                             operators::a(numQubits, s + 1) *
                             operators::adag(numQubits, p + 1) *
                             operators::a(numQubits, r + 1);

          std::unordered_map<spin_op::spin_op_term, std::complex<double>> terms;
          twoElectron.for_each_term([&](spin_op &term) {
            auto coeff = term.get_coefficient();
            if (std::fabs(coeff.real()) < 1e-12 &&
                std::fabs(coeff.imag()) < 1e-12)
              return;

            if (std::fabs(coeff.real()) < 1e-12)
              terms.insert({term.get_raw_data().first[0],
                            std::complex<double>{0., coeff.imag()}});
          });

          if (!terms.empty())
            pool.push_back(terms);
          rs++;
        }
      }
      pq++;
    }
  }

  pq = 0;
  for (auto p : alphaOrbs) {
    for (auto q : betaOrbs) {

      int rs = 0;
      for (auto r : alphaOrbs) {
        for (auto s : betaOrbs) {

          if (pq < rs)
            continue;

          auto twoElectron =
              operators::adag(numQubits, r) * operators::a(numQubits, p) *
                  operators::adag(numQubits, s) * operators::a(numQubits, q) -
              operators::adag(numQubits, q) * operators::a(numQubits, s) *
                  operators::adag(numQubits, p) * operators::a(numQubits, r);
          if (p > q)
            continue;

          twoElectron += operators::adag(numQubits, s - 1) *
                             operators::a(numQubits, q - 1) *
                             operators::adag(numQubits, r + 1) *
                             operators::a(numQubits, p + 1) -
                         operators::adag(numQubits, p + 1) *
                             operators::a(numQubits, r + 1) *
                             operators::adag(numQubits, q - 1) *
                             operators::a(numQubits, s - 1);
          std::unordered_map<spin_op::spin_op_term, std::complex<double>> terms;
          twoElectron.for_each_term([&](spin_op &term) {
            auto coeff = term.get_coefficient();
            if (std::fabs(coeff.real()) < 1e-12 &&
                std::fabs(coeff.imag()) < 1e-12)
              return;

            if (std::fabs(coeff.real()) < 1e-12)
              terms.insert({term.get_raw_data().first[0],
                            std::complex<double>{0., coeff.imag()}});
          });
          if (!terms.empty())
            pool.push_back(terms);
          rs++;
        }
      }
      pq++;
    }
  }

  return pool;
}

} // namespace cudaq