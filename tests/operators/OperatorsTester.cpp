/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cmath>
#include <fstream>
#include <gtest/gtest.h>

#include "cudaq.h"
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

TEST(OperatorsTester, checkIntegrals) {
  cudaq::operators::one_body_integrals hpq(h2_hpq_data.data(), {4, 4});
  cudaq::operators::two_body_integrals hpqrs(h2_hpqrs_data.data(),
                                             {4, 4, 4, 4});
  EXPECT_EQ(2, hpq.shape.size());
  EXPECT_EQ(4, hpq.shape[0]);
  EXPECT_EQ(4, hpq.shape[1]);
  EXPECT_NEAR(hpq(0, 0).real(), -1.2488, 1e-3);
  EXPECT_NEAR(hpq(1, 1).real(), -1.2488, 1e-3);
  EXPECT_NEAR(hpq(2, 2).real(), -.47967, 1e-3);
  EXPECT_NEAR(hpq(3, 3).real(), -.47967, 1e-3);

  EXPECT_EQ(4, hpqrs.shape.size());
  for (int i = 0; i < 4; i++)
    EXPECT_EQ(4, hpqrs.shape[i]);
  EXPECT_NEAR(hpqrs(0, 0, 0, 0).real(), 0.3366719725032414, 1e-3);
  EXPECT_NEAR(hpqrs(0, 0, 2, 2).real(), 0.0908126657382825, 1e-3);
  EXPECT_NEAR(hpqrs(0, 1, 1, 0).real(), 0.3366719725032414, 1e-3);
  EXPECT_NEAR(hpqrs(0, 1, 3, 2).real(), 0.0908126657382825, 1e-3);
  EXPECT_NEAR(hpqrs(0, 2, 0, 2).real(), 0.09081266573828267, 1e-3);
  EXPECT_NEAR(hpqrs(0, 2, 2, 0).real(), 0.33121364716348484, 1e-3);
  EXPECT_NEAR(hpqrs(0, 3, 1, 2).real(), 0.09081266573828267, 1e-3);
  EXPECT_NEAR(hpqrs(0, 3, 3, 0).real(), 0.33121364716348484, 1e-3);
  EXPECT_NEAR(hpqrs(1, 0, 0, 1).real(), 0.3366719725032414, 1e-3);
  EXPECT_NEAR(hpqrs(1, 0, 2, 3).real(), 0.0908126657382825, 1e-3);
  EXPECT_NEAR(hpqrs(1, 1, 1, 1).real(), 0.3366719725032414, 1e-3);
  EXPECT_NEAR(hpqrs(1, 1, 3, 3).real(), 0.0908126657382825, 1e-3);
  EXPECT_NEAR(hpqrs(1, 2, 0, 3).real(), 0.09081266573828267, 1e-3);
  EXPECT_NEAR(hpqrs(1, 2, 2, 1).real(), 0.33121364716348484, 1e-3);
  EXPECT_NEAR(hpqrs(1, 3, 1, 3).real(), 0.09081266573828267, 1e-3);
  EXPECT_NEAR(hpqrs(1, 3, 3, 1).real(), 0.33121364716348484, 1e-3);
  EXPECT_NEAR(hpqrs(2, 0, 0, 2).real(), 0.3312136471634851, 1e-3);
  EXPECT_NEAR(hpqrs(2, 0, 2, 0).real(), 0.09081266573828246, 1e-3);
  EXPECT_NEAR(hpqrs(2, 1, 1, 2).real(), 0.3312136471634851, 1e-3);
  EXPECT_NEAR(hpqrs(2, 1, 3, 0).real(), 0.09081266573828246, 1e-3);
  EXPECT_NEAR(hpqrs(2, 2, 0, 0).real(), 0.09081266573828264, 1e-3);
  EXPECT_NEAR(hpqrs(2, 2, 2, 2).real(), 0.34814578499360427, 1e-3);
  EXPECT_NEAR(hpqrs(2, 3, 1, 0).real(), 0.09081266573828264, 1e-3);
  EXPECT_NEAR(hpqrs(2, 3, 3, 2).real(), 0.34814578499360427, 1e-3);
  EXPECT_NEAR(hpqrs(3, 0, 0, 3).real(), 0.3312136471634851, 1e-3);
  EXPECT_NEAR(hpqrs(3, 0, 2, 1).real(), 0.09081266573828246, 1e-3);
  EXPECT_NEAR(hpqrs(3, 1, 1, 3).real(), 0.3312136471634851, 1e-3);
  EXPECT_NEAR(hpqrs(3, 1, 3, 1).real(), 0.09081266573828246, 1e-3);
  EXPECT_NEAR(hpqrs(3, 2, 0, 1).real(), 0.09081266573828264, 1e-3);
  EXPECT_NEAR(hpqrs(3, 2, 2, 3).real(), 0.34814578499360427, 1e-3);
  EXPECT_NEAR(hpqrs(3, 3, 1, 1).real(), 0.09081266573828264, 1e-3);
  EXPECT_NEAR(hpqrs(3, 3, 3, 3).real(), 0.34814578499360427, 1e-3);
}

TEST(OperatorsTester, checkFermionOp) {
  using namespace cudaq::operators;
  fermion_op fermionOp(4, 0.7080240981000804);
  fermionOp.hpq.set_data(h2_hpq_data);
  fermionOp.hpqrs.set_data(h2_hpqrs_data);

  EXPECT_NEAR(fermionOp.constant, 0.7080240981000804, 1e-3);
  EXPECT_EQ(2, fermionOp.hpq.shape.size());
  EXPECT_EQ(4, fermionOp.hpq.shape[0]);
  EXPECT_EQ(4, fermionOp.hpq.shape[1]);
  EXPECT_NEAR(fermionOp.hpq(0, 0).real(), -1.2488, 1e-3);
  EXPECT_NEAR(fermionOp.hpq(1, 1).real(), -1.2488, 1e-3);
  EXPECT_NEAR(fermionOp.hpq(2, 2).real(), -.47967, 1e-3);
  EXPECT_NEAR(fermionOp.hpq(3, 3).real(), -.47967, 1e-3);

  EXPECT_EQ(4, fermionOp.hpqrs.shape.size());
  for (int i = 0; i < 4; i++)
    EXPECT_EQ(4, fermionOp.hpqrs.shape[i]);
  EXPECT_NEAR(fermionOp.hpqrs(0, 0, 0, 0).real(), 0.3366719725032414, 1e-3);
  EXPECT_NEAR(fermionOp.hpqrs(0, 0, 2, 2).real(), 0.0908126657382825, 1e-3);
  EXPECT_NEAR(fermionOp.hpqrs(0, 1, 1, 0).real(), 0.3366719725032414, 1e-3);
  EXPECT_NEAR(fermionOp.hpqrs(0, 1, 3, 2).real(), 0.0908126657382825, 1e-3);
  EXPECT_NEAR(fermionOp.hpqrs(0, 2, 0, 2).real(), 0.09081266573828267, 1e-3);
  EXPECT_NEAR(fermionOp.hpqrs(0, 2, 2, 0).real(), 0.33121364716348484, 1e-3);
  EXPECT_NEAR(fermionOp.hpqrs(0, 3, 1, 2).real(), 0.09081266573828267, 1e-3);
  EXPECT_NEAR(fermionOp.hpqrs(0, 3, 3, 0).real(), 0.33121364716348484, 1e-3);
  EXPECT_NEAR(fermionOp.hpqrs(1, 0, 0, 1).real(), 0.3366719725032414, 1e-3);
  EXPECT_NEAR(fermionOp.hpqrs(1, 0, 2, 3).real(), 0.0908126657382825, 1e-3);
  EXPECT_NEAR(fermionOp.hpqrs(1, 1, 1, 1).real(), 0.3366719725032414, 1e-3);
  EXPECT_NEAR(fermionOp.hpqrs(1, 1, 3, 3).real(), 0.0908126657382825, 1e-3);
  EXPECT_NEAR(fermionOp.hpqrs(1, 2, 0, 3).real(), 0.09081266573828267, 1e-3);
  EXPECT_NEAR(fermionOp.hpqrs(1, 2, 2, 1).real(), 0.33121364716348484, 1e-3);
  EXPECT_NEAR(fermionOp.hpqrs(1, 3, 1, 3).real(), 0.09081266573828267, 1e-3);
  EXPECT_NEAR(fermionOp.hpqrs(1, 3, 3, 1).real(), 0.33121364716348484, 1e-3);
  EXPECT_NEAR(fermionOp.hpqrs(2, 0, 0, 2).real(), 0.3312136471634851, 1e-3);
  EXPECT_NEAR(fermionOp.hpqrs(2, 0, 2, 0).real(), 0.09081266573828246, 1e-3);
  EXPECT_NEAR(fermionOp.hpqrs(2, 1, 1, 2).real(), 0.3312136471634851, 1e-3);
  EXPECT_NEAR(fermionOp.hpqrs(2, 1, 3, 0).real(), 0.09081266573828246, 1e-3);
  EXPECT_NEAR(fermionOp.hpqrs(2, 2, 0, 0).real(), 0.09081266573828264, 1e-3);
  EXPECT_NEAR(fermionOp.hpqrs(2, 2, 2, 2).real(), 0.34814578499360427, 1e-3);
  EXPECT_NEAR(fermionOp.hpqrs(2, 3, 1, 0).real(), 0.09081266573828264, 1e-3);
  EXPECT_NEAR(fermionOp.hpqrs(2, 3, 3, 2).real(), 0.34814578499360427, 1e-3);
  EXPECT_NEAR(fermionOp.hpqrs(3, 0, 0, 3).real(), 0.3312136471634851, 1e-3);
  EXPECT_NEAR(fermionOp.hpqrs(3, 0, 2, 1).real(), 0.09081266573828246, 1e-3);
  EXPECT_NEAR(fermionOp.hpqrs(3, 1, 1, 3).real(), 0.3312136471634851, 1e-3);
  EXPECT_NEAR(fermionOp.hpqrs(3, 1, 3, 1).real(), 0.09081266573828246, 1e-3);
  EXPECT_NEAR(fermionOp.hpqrs(3, 2, 0, 1).real(), 0.09081266573828264, 1e-3);
  EXPECT_NEAR(fermionOp.hpqrs(3, 2, 2, 3).real(), 0.34814578499360427, 1e-3);
  EXPECT_NEAR(fermionOp.hpqrs(3, 3, 1, 1).real(), 0.09081266573828264, 1e-3);
  EXPECT_NEAR(fermionOp.hpqrs(3, 3, 3, 3).real(), 0.34814578499360427, 1e-3);
}

TEST(OperatorsTester, checkJordanWigner) {

  using namespace cudaq::operators;
  fermion_op fermionOp(4, 0.7080240981000804);
  fermionOp.hpq.set_data(h2_hpq_data);
  fermionOp.hpqrs.set_data(h2_hpqrs_data);

  auto jw = fermion_compiler::get("jordan_wigner");
  auto spin = jw->generate(fermionOp);
  auto groundEnergy = spin.to_matrix().minimal_eigenvalue().real();
  EXPECT_NEAR(groundEnergy, -1.137, 1e-3);
}

TEST(OperatorsTester, checkMolecule) {
  cudaq::operators::molecular_geometry geometry{{"H", {0., 0., 0.}},
                                                {"H", {0., 0., .7474}}};
  auto molecule = cudaq::operators::create_molecule(
      geometry, "sto-3g", 0, 0, {.casci = true, .verbose = true});

  molecule.hamiltonian.dump();
  EXPECT_NEAR(molecule.energies["fci_energy"], -1.137, 1e-3);
  EXPECT_NEAR(molecule.energies["hf_energy"], -1.1163255644, 1e-3);
  EXPECT_EQ(molecule.n_electrons, 2);
  EXPECT_EQ(molecule.n_orbitals, 2);

  EXPECT_NEAR(molecule.fermionOperator.constant, 0.7080240981000804, 1e-3);
  EXPECT_EQ(2, molecule.fermionOperator.hpq.shape.size());
  EXPECT_EQ(4, molecule.fermionOperator.hpq.shape[0]);
  EXPECT_EQ(4, molecule.fermionOperator.hpq.shape[1]);
  EXPECT_NEAR(molecule.fermionOperator.hpq(0, 0).real(), -1.2488, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpq(1, 1).real(), -1.2488, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpq(2, 2).real(), -.47967, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpq(3, 3).real(), -.47967, 1e-3);
  EXPECT_EQ(4, molecule.fermionOperator.hpqrs.shape.size());
  for (int i = 0; i < 4; i++)
    EXPECT_EQ(4, molecule.fermionOperator.hpqrs.shape[i]);
  EXPECT_NEAR(molecule.fermionOperator.hpqrs(0, 0, 0, 0).real(),
              0.3366719725032414, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpqrs(0, 0, 2, 2).real(),
              0.0908126657382825, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpqrs(0, 1, 1, 0).real(),
              0.3366719725032414, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpqrs(0, 1, 3, 2).real(),
              0.0908126657382825, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpqrs(0, 2, 0, 2).real(),
              0.09081266573828267, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpqrs(0, 2, 2, 0).real(),
              0.33121364716348484, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpqrs(0, 3, 1, 2).real(),
              0.09081266573828267, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpqrs(0, 3, 3, 0).real(),
              0.33121364716348484, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpqrs(1, 0, 0, 1).real(),
              0.3366719725032414, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpqrs(1, 0, 2, 3).real(),
              0.0908126657382825, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpqrs(1, 1, 1, 1).real(),
              0.3366719725032414, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpqrs(1, 1, 3, 3).real(),
              0.0908126657382825, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpqrs(1, 2, 0, 3).real(),
              0.09081266573828267, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpqrs(1, 2, 2, 1).real(),
              0.33121364716348484, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpqrs(1, 3, 1, 3).real(),
              0.09081266573828267, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpqrs(1, 3, 3, 1).real(),
              0.33121364716348484, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpqrs(2, 0, 0, 2).real(),
              0.3312136471634851, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpqrs(2, 0, 2, 0).real(),
              0.09081266573828246, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpqrs(2, 1, 1, 2).real(),
              0.3312136471634851, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpqrs(2, 1, 3, 0).real(),
              0.09081266573828246, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpqrs(2, 2, 0, 0).real(),
              0.09081266573828264, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpqrs(2, 2, 2, 2).real(),
              0.34814578499360427, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpqrs(2, 3, 1, 0).real(),
              0.09081266573828264, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpqrs(2, 3, 3, 2).real(),
              0.34814578499360427, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpqrs(3, 0, 0, 3).real(),
              0.3312136471634851, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpqrs(3, 0, 2, 1).real(),
              0.09081266573828246, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpqrs(3, 1, 1, 3).real(),
              0.3312136471634851, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpqrs(3, 1, 3, 1).real(),
              0.09081266573828246, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpqrs(3, 2, 0, 1).real(),
              0.09081266573828264, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpqrs(3, 2, 2, 3).real(),
              0.34814578499360427, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpqrs(3, 3, 1, 1).real(),
              0.09081266573828264, 1e-3);
  EXPECT_NEAR(molecule.fermionOperator.hpqrs(3, 3, 3, 3).real(),
              0.34814578499360427, 1e-3);
}

TEST(OperatorsTester, checkH2OActiveSpace) {
  std::string contents = R"#(3

O 0.1173 0.0 0.0
H -0.4691 0.7570 0.0
H -0.4691 -0.7570 0.0
)#";

  {
    std::ofstream out(".tmpH2O.xyz");
    out << contents;
  }

  auto geometry = cudaq::operators::molecular_geometry::from_xyz(".tmpH2O.xyz");
  std::remove(".tmpH2O.xyz");

  auto molecule = cudaq::operators::create_molecule(
      geometry, "631g", 0, 0,
      {.nele_cas = 6, .norb_cas = 6, .ccsd = true, .verbose = true});

  // molecule.hamiltonian.dump();
  EXPECT_EQ(molecule.n_electrons, 6);
  EXPECT_EQ(molecule.n_orbitals, 6);
}

TEST(OperatorsTester, checkOneParticleOp) {
  auto op = cudaq::operators::one_particle_op(4, 1, 1);
  EXPECT_EQ(op.num_terms(), 2);
  op.dump();
}