import cudaq, cudaqlib, numpy as np

# Define the molecule
geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
molecule = cudaqlib.operators.create_molecule(geometry,
                                              'sto-3g',
                                              0,
                                              0)

# Get the number of qubits
hamiltonian = molecule.hamiltonian

# Get the number of qubits
numQubits = molecule.hamiltonian.get_qubit_count()

# Create the operator pool
# FIXME needs to better incorporate thetas.
pool = cudaqlib.gse.get_operator_pool('uccsd', num_qubits=4, num_electrons=2)
operatorCoeffs = [
        0.003125, -0.003125, 0.00625, -0.00625, 0.0125, -0.0125, 0.025, -0.025,
        0.05, -0.05, 0.1, -0.1
    ]
assert len(pool) == len(operatorCoeffs
                        )
# Need an initial state
@cudaq.kernel
def init(q: cudaq.qview):
    x(q[0])
    x(q[1])


# Define the GQE cost function
def cost(sampledPoolOperations : list): 
    """
    Cost should take operator pool indices and 
    return the associated cost. For the chemistry 
    example, we'll take uccsd pool indices and return 
    cudaq observe result
    """
    asWords = [cudaq.pauli_word(op.to_string(False)) for op in sampledPoolOperations]

    @cudaq.kernel
    def kernel(numQubits: int, coeffs: list[float],
               words: list[cudaq.pauli_word]):
        q = cudaq.qvector(numQubits)
        init(q)
        for i, word in enumerate(words):
            exp_pauli(coeffs[i], q, word)

    return cudaq.observe(kernel, molecule.hamiltonian, numQubits, operatorCoeffs, asWords).expectation()


minE, optimPoolOps = cudaqlib.gqe(cost, pool)
print(f'Ground Energy = {minE}')
print('Ansatz Ops')
for idx in optimPoolOps: print(pool[idx].to_string(False))