# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os

import pytest, pathlib
import numpy as np

import cudaq, cudaqlib

currentPath = pathlib.Path(__file__).parent.resolve()


def test_operators():
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule = cudaqlib.operators.create_molecule(geometry,
                                                  'sto-3g',
                                                  0,
                                                  0,
                                                  verbose=True,
                                                  casci=True)
    print(molecule.hamiltonian.to_string())
    print(molecule.energies)
    assert np.isclose(-1.11, molecule.energies['hf_energy'], atol=1e-2)
    assert np.isclose(-1.13, molecule.energies['fci_energy'], atol=1e-2)
    minE = molecule.hamiltonian.to_matrix().minimal_eigenvalue()
    assert np.isclose(-1.13, minE, atol=1e-2)



def test_from_xyz_filename():
    molecule = cudaqlib.operators.create_molecule(str(currentPath) +
                                                  '/resources/LiH.xyz',
                                                  'sto-3g',
                                                  0,
                                                  0,
                                                  verbose=True)
    print(molecule.energies)
    print(molecule.n_orbitals)
    print(molecule.n_electrons)
    assert molecule.n_orbitals == 6 
    assert molecule.n_electrons == 4

def test_jordan_wigner():

    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule = cudaqlib.operators.create_molecule(geometry,
                                                  'sto-3g',
                                                  0,
                                                  0,
                                                  verbose=True,
                                                  casci=True)
    op = cudaqlib.operators.jordan_wigner(molecule.fermion_operator)
    print(op.to_string())
    assert molecule.hamiltonian == op
    hpq = np.array(molecule.fermion_operator.hpq)
    hpqrs = np.array(molecule.fermion_operator.hpqrs)
    hpqJw = cudaqlib.operators.jordan_wigner(hpq, molecule.energies['nuclear_energy'])
    hpqrsJw = cudaqlib.operators.jordan_wigner(hpqrs)
    op2 = hpqJw + hpqrsJw 
    assert op2 == molecule.hamiltonian


def test_chemistry_operators():

    @cudaq.kernel
    def ansatz(thetas: list[float]):
        q = cudaq.qvector(4)
        x(q[0])
        x(q[1])
        exp_pauli(thetas[0], q, 'XXXY')

    # Define the molecule
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule = cudaqlib.operators.create_molecule(geometry,
                                                  'sto-3g',
                                                  0,
                                                  0,
                                                  verbose=True)

    # Run VQE
    energy, params, all_data = cudaqlib.gse.vqe(ansatz,
                                                molecule.hamiltonian, [0.],
                                                optimizer='cobyla')
    assert np.isclose(-1.137, energy, atol=1e-2)
