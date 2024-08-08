import os

import numpy as np
import cudaq, cudaqlib


def test_H2O_uccsd_natorb():
    
    geometry = [('O', (0.1173,0.0,0.0)), ('H', (-0.4691,0.7570,0.0)), ('H', (-0.4691,-0.7570,0.0))]
    molecule = cudaqlib.operators.create_molecule(geometry,
                                                '631g',
                                                0,
                                                0,
                                                type='gas_phase',
                                                MP2=True,
                                                natorb=True,
                                                ccsd=True,
                                                integrals_natorb=True,
                                                nele_cas=4,
                                                norb_cas=4,
                                                verbose=True)

    
    params_num=cudaqlib.kernels.uccsd_num_parameters(molecule.n_electrons,2*molecule.n_orbitals)
    
    #print(molecule.n_electrons)
    #print(molecule.n_orbitals)
    #print(params_num)
    
    qubits_num=2*molecule.n_orbitals
    n_elec=molecule.n_electrons
    
    # Define the state preparation ansatz
    @cudaq.kernel
    def ansatz(thetas: list[float]):
        q = cudaq.qvector(qubits_num)
        for i in range(n_elec):
            x(q[i])
        cudaqlib.kernels.uccsd(q, thetas, n_elec, qubits_num)
    
    np.random.seed(42)
    x0 = np.random.normal(0, 1, params_num)
    
    # Run VQE
    energy, params, all_data = cudaqlib.gse.vqe(ansatz,
                                            molecule.hamiltonian, x0.tolist(),
                                            optimizer='cobyla')
    
    
    print(energy)
    
    
def test_H2O_uccsd():
    
    geometry = [('O', (0.1173,0.0,0.0)), ('H', (-0.4691,0.7570,0.0)), ('H', (-0.4691,-0.7570,0.0))]
    molecule = cudaqlib.operators.create_molecule(geometry,
                                                '631g',
                                                0,
                                                0,
                                                type='gas_phase',
                                                MP2=True,
                                                ccsd=True,
                                                nele_cas=4,
                                                norb_cas=4,
                                                verbose=True)

    
    params_num=cudaqlib.kernels.uccsd_num_parameters(molecule.n_electrons,2*molecule.n_orbitals)
    
    qubits_num=2*molecule.n_orbitals
    n_elec=molecule.n_electrons
    
    # Define the state preparation ansatz
    @cudaq.kernel
    def ansatz(thetas: list[float]):
        q = cudaq.qvector(qubits_num)
        for i in range(n_elec):
            x(q[i])
        cudaqlib.kernels.uccsd(q, thetas, n_elec, qubits_num)
    
    np.random.seed(42)
    x0 = np.random.normal(0, 1, params_num)
    
    # Run VQE
    energy, params, all_data = cudaqlib.gse.vqe(ansatz,
                                            molecule.hamiltonian, x0.tolist(),
                                            optimizer='cobyla')
    
    
    print(energy)
    
test_H2O_uccsd_natorb()
test_H2O_uccsd()