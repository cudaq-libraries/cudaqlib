import cudaq, cudaqlib

cudaq.mpi.initialize()
cudaq.set_target('nvidia', option='mqpu')

# Define the molecule
geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
molecule = cudaqlib.operators.create_molecule(geometry, 'sto-3g', 0, 0)

# Get the number of qubits
hamiltonian = molecule.hamiltonian

# Get the number of qubits
numQubits = molecule.hamiltonian.get_qubit_count()

# Create the operator pool
pool = cudaqlib.gse.get_operator_pool('uccsd',
                                      num_qubits=4,
                                      num_electrons=2,
                                      operator_coeffs=[
                                          0.003125, -0.003125, 0.00625,
                                          -0.00625, 0.0125, -0.0125, 0.025,
                                          -0.025, 0.05, -0.05, 0.1, -0.1
                                      ])

# Need an initial state
@cudaq.kernel
def init(q: cudaq.qview):
    x(q[0])
    x(q[1])

@cudaq.kernel
def kernel(numQubits: int, coeffs: list[float],
            words: list[cudaq.pauli_word]):
    q = cudaq.qvector(numQubits)
    init(q)
    for i, word in enumerate(words):
        exp_pauli(coeffs[i], q, word)

# Define the GQE cost function
def cost(sampledPoolOperations: list, qpu_id : int = 0):
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
    operatorCoeffs = [
        op.get_coefficient().real for op in sampledPoolOperations
    ]

    handle = cudaq.observe_async(kernel,
                               hamiltonian,
                               numQubits,
                               operatorCoeffs,
                               asWords,
                               qpu_id=qpu_id)
    
    # If in async mode, you return the async handle and the 
    # functor that extracts the scalar you care about 
    return handle, lambda res: res.get().expectation()

minE, optimPoolOps = cudaqlib.gqe(cost, pool, max_iters=20)

if cudaq.mpi.rank() == 0:
    print(f'Ground Energy = {minE}')
    print('Ansatz Ops')
    for idx in optimPoolOps:
        print(pool[idx].get_coefficient().real, pool[idx].to_string(False))

cudaq.mpi.finalize()