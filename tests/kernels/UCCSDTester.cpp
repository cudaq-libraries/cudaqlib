/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cmath>
#include <gtest/gtest.h>

#include "cudaq.h"
#include "cudaqlib/gse.h"
#include "cudaqlib/kernels/uccsd.h"
#include "cudaqlib/operators.h"

std::vector<std::complex<double>> h2_hpq_data{
    -1.24884680e+00, 0.00000000e+00,  -9.24110683e-17, 0.00000000e+00,
    0.00000000e+00,  -1.24884680e+00, 0.00000000e+00,  -9.24110683e-17,
    -2.68142410e-16, 0.00000000e+00,  -4.79677813e-01, 0.00000000e+00,
    0.00000000e+00,  -2.68142410e-16, 0.00000000e+00,  -4.79677813e-01};

std::vector<std::complex<double>> h2_hpqrs_data{0.3366719725032414,
                                                0.0,
                                                5.898059818321144e-17,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                5.551115123125783e-17,
                                                0.0,
                                                0.0908126657382825,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.3366719725032414,
                                                0.0,
                                                5.898059818321144e-17,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                5.551115123125783e-17,
                                                0.0,
                                                0.0908126657382825,
                                                0.0,
                                                9.71445146547012e-17,
                                                0.0,
                                                0.09081266573828267,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.33121364716348484,
                                                0.0,
                                                5.551115123125783e-17,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                9.71445146547012e-17,
                                                0.0,
                                                0.09081266573828267,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.33121364716348484,
                                                0.0,
                                                5.551115123125783e-17,
                                                0.0,
                                                0.0,
                                                0.3366719725032414,
                                                0.0,
                                                5.898059818321144e-17,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                5.551115123125783e-17,
                                                0.0,
                                                0.0908126657382825,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.3366719725032414,
                                                0.0,
                                                5.898059818321144e-17,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                5.551115123125783e-17,
                                                0.0,
                                                0.0908126657382825,
                                                0.0,
                                                9.71445146547012e-17,
                                                0.0,
                                                0.09081266573828267,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.33121364716348484,
                                                0.0,
                                                5.551115123125783e-17,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                9.71445146547012e-17,
                                                0.0,
                                                0.09081266573828267,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.33121364716348484,
                                                0.0,
                                                5.551115123125783e-17,
                                                7.979727989493313e-17,
                                                0.0,
                                                0.3312136471634851,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.09081266573828246,
                                                0.0,
                                                1.1102230246251565e-16,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                7.979727989493313e-17,
                                                0.0,
                                                0.3312136471634851,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.09081266573828246,
                                                0.0,
                                                1.1102230246251565e-16,
                                                0.0,
                                                0.09081266573828264,
                                                0.0,
                                                8.326672684688674e-17,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                8.326672684688674e-17,
                                                0.0,
                                                0.34814578499360427,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.09081266573828264,
                                                0.0,
                                                8.326672684688674e-17,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                8.326672684688674e-17,
                                                0.0,
                                                0.34814578499360427,
                                                0.0,
                                                0.0,
                                                7.979727989493313e-17,
                                                0.0,
                                                0.3312136471634851,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.09081266573828246,
                                                0.0,
                                                1.1102230246251565e-16,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                7.979727989493313e-17,
                                                0.0,
                                                0.3312136471634851,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.09081266573828246,
                                                0.0,
                                                1.1102230246251565e-16,
                                                0.0,
                                                0.09081266573828264,
                                                0.0,
                                                8.326672684688674e-17,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                8.326672684688674e-17,
                                                0.0,
                                                0.34814578499360427,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.09081266573828264,
                                                0.0,
                                                8.326672684688674e-17,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                                8.326672684688674e-17,
                                                0.0,
                                                0.34814578499360427};

TEST(UCCSDTester, checkUCCSDAnsatz) {
  cudaq::operators::molecular_geometry geometry{{"H", {0., 0., 0.}},
                                                {"H", {0., 0., .7474}}};
  auto molecule = cudaq::operators::create_molecule(geometry, "sto-3g", 0, 0,
                                                    {.verbose = true});

  auto numElectrons = molecule.n_electrons;
  auto numQubits = molecule.n_orbitals * 2;

  // EXPECT_NEAR(molecule.fci_energy, -1.137, 1e-3);
  EXPECT_NEAR(molecule.energies["hf_energy"], -1.1163255644, 1e-3);
  EXPECT_EQ(numElectrons, 2);
  EXPECT_EQ(numQubits, 4);

  auto ansatz = [numElectrons, numQubits](std::vector<double> thetas) __qpu__ {
    cudaq::qvector q(numQubits);
    for (auto i : cudaq::range(numElectrons))
      x(q[i]);
    cudaq::uccsd(q, thetas, numElectrons);
  };

  auto [singlesAlpha, singlesBeta, doublesMixed, doublesAlpha, doublesBeta] =
      cudaq::get_uccsd_excitations(numElectrons, numQubits);
  EXPECT_TRUE(doublesAlpha.empty());
  EXPECT_TRUE(doublesBeta.empty());
  EXPECT_TRUE(singlesAlpha.size() == 1);
  EXPECT_EQ(singlesAlpha[0][0], 0);
  EXPECT_EQ(singlesAlpha[0][1], 2);
  EXPECT_EQ(singlesBeta[0][0], 1);
  EXPECT_EQ(singlesBeta[0][1], 3);
  EXPECT_EQ(doublesMixed[0][0], 0);
  EXPECT_EQ(doublesMixed[0][1], 1);
  EXPECT_EQ(doublesMixed[0][2], 3);
  EXPECT_EQ(doublesMixed[0][3], 2);
  EXPECT_TRUE(singlesBeta.size() == 1);
  EXPECT_TRUE(doublesMixed.size() == 1);

  auto numParams = cudaq::uccsd_num_parameters(numElectrons, numQubits);
  std::vector<double> init{-2., -2., -2.};

  cudaq::optim::cobyla optimizer;
  auto result = cudaq::gse::vqe(ansatz, molecule.hamiltonian, optimizer, init,
                                {.verbose = true});
  EXPECT_NEAR(result.energy, -1.137, 1e-3);

  // test that we can use the builder
  {
    auto [kernel, thetas] = cudaq::make_kernel<std::vector<double>>();
    auto q = kernel.qalloc(numQubits);
    for (auto i : cudaq::range(numElectrons))
      kernel.x(q[i]);
    cudaq::uccsd(kernel, q, thetas, numElectrons, numQubits);

    result = cudaq::gse::vqe(kernel, molecule.hamiltonian, optimizer, init,
                             {.verbose = true});
    EXPECT_NEAR(result.energy, -1.137, 1e-3);
  }
}

TEST(UCCSDTester, checkOperatorPool) {
  auto pool = cudaq::operator_pool::get("uccsd");
  auto ops = pool->generate({{"num-qubits", 4}, {"num-electrons", 2}});

  for (auto o : ops)
    o.dump();
}
