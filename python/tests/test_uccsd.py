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

def test_uccsd_vqe():

    # Define the molecule
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule = cudaqlib.operators.create_molecule(geometry,
                                                'sto-3g',
                                                0,
                                                0,
                                                verbose=True)

    # Define the state preparation ansatz
    @cudaq.kernel
    def ansatz(thetas: list[float]):
        q = cudaq.qvector(4)
        for i in range(2):
            x(q[i])
        cudaqlib.kernels.uccsd(q, thetas, 2, 4)


    # Run VQE
    energy, params, all_data = cudaqlib.gse.vqe(ansatz,
                                            molecule.hamiltonian, [-2., -2., -2.],
                                            optimizer='cobyla')

    assert np.isclose(-1.137, energy, atol=1e-3)
