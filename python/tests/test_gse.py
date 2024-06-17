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

import cudaq
from cudaq import spin

import cudaqlib


def test_gse_vqe():

    @cudaq.kernel
    def ansatz(theta: float):
        q = cudaq.qvector(2)
        x(q[0])
        ry(theta, q[1])
        x.ctrl(q[1], q[0])

    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    # Can specify optimizer and gradient
    energy, params, all_data = cudaqlib.gse.vqe(lambda thetas: ansatz(thetas[0]),
                                              hamiltonian, [0.],
                                              optimizer='lbfgs',
                                              gradient='parameter_shift')
    assert np.isclose(-1.74, energy, atol=1e-2)
    all_data[0].result.dump()

    # For gradient-based optimizer, can pick up default gradient (parameter_shift)
    energy, params, all_data = cudaqlib.gse.vqe(lambda thetas: ansatz(thetas[0]),
                                              hamiltonian, [0.],
                                              optimizer='lbfgs',
                                              verbose=True)
    assert np.isclose(-1.74, energy, atol=1e-2)

    # Can pick up default optimizer (cobyla)
    energy, params, all_data = cudaqlib.gse.vqe(lambda thetas: ansatz(thetas[0]),
                                              hamiltonian, [0.],
                                              verbose=True)
    assert np.isclose(-1.74, energy, atol=1e-2)

    cudaq.set_random_seed(22)

    # Can pick up default optimizer (cobyla)
    energy, params, all_data = cudaqlib.gse.vqe(lambda thetas: ansatz(thetas[0]),
                                              hamiltonian, [0.],
                                              verbose=True,
                                              shots=10000,
                                              max_iterations=10)
    assert energy > -2 and energy < -1.5
    print(energy)
    all_data[0].result.dump()
    counts = all_data[0].result.counts()
    assert 5 == len(counts.register_names)
    assert 4 == len(counts.get_register_counts('XX'))
    assert 4 == len(counts.get_register_counts('YY'))
    assert 1 == len(counts.get_register_counts('ZI'))
    assert 1 == len(counts.get_register_counts('IZ'))
