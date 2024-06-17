#pragma once

#include "../operator_pool.h"

namespace cudaq::gse {

// adapted from
// https://github.com/mayhallgroup/adapt-vqe/blob/master/src/operator_pools.py

class spin_complement_gsd : public operator_pool {
protected:
  std::int64_t numOrbitals;

public:
  spin_complement_gsd(std::int64_t q) : numOrbitals(q) {}
  std::vector<spin_op> generate() const override {
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
            fermion::adag(numQubits, q) * fermion::a(numQubits, p) -
            fermion::adag(numQubits, p) * fermion::a(numQubits, q);
        oneElectron +=
            fermion::adag(numQubits, q + 1) * fermion::a(numQubits, p + 1) -
            fermion::adag(numQubits, p + 1) * fermion::a(numQubits, q + 1);

        std::unordered_map<spin_op::spin_op_term, std::complex<double>> terms;
        oneElectron.for_each_term([&](spin_op &term) {
          auto coeff = term.get_coefficient();
          if (std::fabs(coeff.real()) < 1e-12 &&
              std::fabs(coeff.imag()) < 1e-12)
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
                fermion::adag(numQubits, r) * fermion::a(numQubits, p) *
                    fermion::adag(numQubits, s) * fermion::a(numQubits, q) -
                fermion::adag(numQubits, q) * fermion::a(numQubits, s) *
                    fermion::adag(numQubits, p) * fermion::a(numQubits, r);
            twoElectron +=
                fermion::adag(numQubits, r + 1) * fermion::a(numQubits, p + 1) *
                    fermion::adag(numQubits, s + 1) *
                    fermion::a(numQubits, q + 1) -
                fermion::adag(numQubits, q + 1) * fermion::a(numQubits, s + 1) *
                    fermion::adag(numQubits, p + 1) *
                    fermion::a(numQubits, r + 1);

            std::unordered_map<spin_op::spin_op_term, std::complex<double>>
                terms;
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
                fermion::adag(numQubits, r) * fermion::a(numQubits, p) *
                    fermion::adag(numQubits, s) * fermion::a(numQubits, q) -
                fermion::adag(numQubits, q) * fermion::a(numQubits, s) *
                    fermion::adag(numQubits, p) * fermion::a(numQubits, r);
            if (p > q)
              continue;

            twoElectron +=
                fermion::adag(numQubits, s - 1) * fermion::a(numQubits, q - 1) *
                    fermion::adag(numQubits, r + 1) *
                    fermion::a(numQubits, p + 1) -
                fermion::adag(numQubits, p + 1) * fermion::a(numQubits, r + 1) *
                    fermion::adag(numQubits, q - 1) *
                    fermion::a(numQubits, s - 1);
            std::unordered_map<spin_op::spin_op_term, std::complex<double>>
                terms;
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
};
} // namespace cudaq::gse