import cudaq, cudaqlib

# Define the molecule
distance = 1.1621
geometry = [('O', (0., 0., 0.)), ('C', (0., 0., distance)), ('O', (0., 0., 2*distance))]
molecule = cudaqlib.operators.create_molecule(geometry, 'sto-3g', 0, 0, MP2=True, nele_cas=10, norb_cas=9)

# Get the system Hamiltonian
hamiltonian = molecule.hamiltonian

# Get the number of qubits
numQubits = molecule.hamiltonian.get_qubit_count()

# Create the operator pool
pool = cudaqlib.gse.get_operator_pool('uccsd',
                                      num_qubits=numQubits,
                                      num_electrons=10,
                                      operator_coeffs=[0.003125, -0.003125, 0.00625, -0.00625, 0.0125, -0.0125, 0.025, -0.025, 0.05, -0.05, 0.1, -0.1])

# Define Hartree-Fock
@cudaq.kernel
def init(q: cudaq.qview):
    for i in range(10):
        x(q[i])


# Define the GQE cost function
def cost(sampledPoolOperations: list):
    """
    Cost should take operator pool indices and
    return the associated cost. For the chemistry
    example, we'll take uccsd pool indices and return
    cudaq observe result
    """
    # Convert the operator pool elements to cudaq.pauli_words
    asWords = [
        cudaq.pauli_word(op.to_string(False)) for op in sampledPoolOperations
    ]

    # Get the pool coefficients as its own list
    operatorCoeffs = [op.get_coefficient().real for op in sampledPoolOperations]

    @cudaq.kernel
    def kernel(numQubits: int, coeffs: list[float],
            words: list[cudaq.pauli_word]):
        q = cudaq.qvector(numQubits)
        init(q)
        for i, word in enumerate(words):
            exp_pauli(coeffs[i], q, word)

    return cudaq.observe(kernel, molecule.hamiltonian, numQubits,
                         operatorCoeffs, asWords).expectation()


minE, optimPoolOps = cudaqlib.gqe(cost, pool, max_iters=10, energy_offset=184.)
print(f'Ground Energy = {minE}')
print('Ansatz Ops')
for idx in optimPoolOps:
    print(pool[idx].get_coefficient().real, pool[idx].to_string(False))
