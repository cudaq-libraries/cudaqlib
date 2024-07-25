# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os

import pytest
import numpy as np

import cudaq, cudaqlib
from cudaq import spin

def test_operators():
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule = cudaqlib.operators.create_molecule(
        geometry, 'sto-3g', 0, 0, verbose=True, casci=True)
    print(molecule.hamiltonian.to_string())
    print(molecule.energies)
    assert np.isclose(-1.11, molecule.energies['hf_energy'], atol=1e-2)
    assert np.isclose(-1.13, molecule.energies['fci_energy'], atol=1e-2)
    minE = molecule.hamiltonian.to_matrix().minimal_eigenvalue()
    assert np.isclose(-1.13, minE, atol=1e-2)




def test_chemistry_operators():

    @cudaq.kernel 
    def ansatz(thetas:list[float]):
        q = cudaq.qvector(4)
        x(q[0])
        x(q[1])
        exp_pauli(thetas[0], q, 'XXXY')

    # Define the molecule
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule = cudaqlib.operators.create_molecule(
        geometry, 'sto-3g', 0, 0, verbose=True)

    # Run VQE
    energy, params, all_data = cudaqlib.gse.vqe(ansatz,
                                            molecule.hamiltonian, [0.],
                                            optimizer='cobyla')
    assert np.isclose(-1.137, energy, atol=1e-2)
