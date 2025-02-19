import numpy as np
import random
import pyzx as zx
from qiskit.circuit import QuantumCircuit
import qiskit.qasm2
import os 
import sys

from alphatensor_quantum.src.experiment.qasm_to_todd import qasm_to_tfc, qasm_from_todd_output

def generate_waring_matrix(circuit):
    """
    Generate the Waring decomposition matrix A from a quantum circuit description.
    Following step 7 of appendix C.1 of the AlphaTensor-Quantum paper.
    
    Args:
        circuit (list): List of strings representing gates in the format:
            - 'T(i)' for T gates where i is the qubit index
            - 'CX(i,j)' for CNOT gates where i is control and j is target
    
    Returns:
        numpy.ndarray: The matrix A where A[j,i] = 1 if T gate i appears on qubit j,
        modified according to the CNOT propagation rules.
    """
    # First, count the number of T gates and find the maximum qubit index
    num_t_gates = sum(1 for gate in circuit if gate.startswith('T'))
    max_qubit = -1
    for gate in circuit:
        if gate.startswith('T'):
            qubit = int(gate[2:-1])
            max_qubit = max(max_qubit, qubit)
        elif gate.startswith('CX'):
            # Assumes CX(i,j) with no space between "," and "j"!!!
            control, target = map(int, gate[3:-1].split(','))
            max_qubit = max(max_qubit, control, target)
    
    num_qubits = max_qubit + 1
    
    # Initialize the A matrix with zeros
    A = np.zeros((num_qubits, num_t_gates), dtype=int)
    
    # First pass: Mark T gate positions
    t_gate_counter = 0
    for gate in circuit:
        if gate.startswith('T'):
            qubit = int(gate[2:-1])
            A[qubit, t_gate_counter] = 1
            t_gate_counter += 1
    
    # Second pass: Process CNOT gates
    t_gate_index = 0
    for gate_idx, gate in enumerate(circuit):
        if gate.startswith('T'):
            t_gate_index += 1
        elif gate.startswith('CX'):
            control, target = map(int, gate[3:-1].split(','))
            # For all T gates to the right of this CNOT
            for i in range(t_gate_index, num_t_gates):
                # Add control row to target row (mod 2)
                A[control, i] = (A[control, i] + A[target, i]) % 2
    
    return A

def generate_random_circuit(num_qubits, num_t_gates, circuit_size):
    """
    Generate a random quantum circuit with specified parameters.
    
    Args:
        num_qubits (int): Number of qubits in the circuit
        num_t_gates (int): Number of T gates to include
        circuit_size (int): Total number of gates (T gates + CNOT gates)
    
    Returns:
        list: List of strings representing gates in format 'T(i)' or 'CX(i,j)'
        qc: Quantum circuit in qiskit format
    """
    if circuit_size < num_t_gates:
        raise ValueError("Circuit size must be greater than or equal to number of T gates")
    
    if num_qubits < 2:
        raise ValueError("Need at least 2 qubits for CNOT gates")
        
    # Calculate number of CNOT gates
    num_cnot = circuit_size - num_t_gates
    
    # Create list of all possible gates
    t_gates = [f'T({i})' for i in range(num_qubits)]
    cx_gates = []
    for i in range(num_qubits):
        for j in range(num_qubits):
            if i != j:  # Can't have control and target be the same qubit
                cx_gates.append(f'CX({i},{j})')
    
    # Select random gates
    selected_t = random.choices(t_gates, k=num_t_gates)
    selected_cx = random.choices(cx_gates, k=num_cnot)
    
    # Combine all gates
    all_gates = selected_t + selected_cx
    
    # Shuffle the gates
    random.shuffle(all_gates)

    ## Create qiskit circuit
    qc = QuantumCircuit(num_qubits)
    for gate in all_gates:
        eval('qc.%s' % (gate.lower()))

    return all_gates, qc

def optimize_circuit_pyzx(circuit_qiskit):
    """
    Optimized a Qiskit QuantumCircuit with PyZX.
    
    Args:
        circuit_qiskit (qiskit.QuantumCircuit): quantum circuit input
    
    Returns:
        new_tcount: the number of tcount 
        opt_circ: optimized quantum circuit in qiskit format
    """
    ## Get circuit from qasm
    circuit = zx.Circuit.from_qasm(qiskit.qasm2.dumps(circuit_qiskit))

    ## Optimize circuit with pyzx
    g = circuit.to_graph()
    g = zx.teleport_reduce(g)
    c_opt = zx.Circuit.from_graph(g)

    ## Convert back to circuit and count t gate
    opt_circ = qiskit.qasm2.loads(c_opt.to_qasm())

    return opt_circ

def optimize_circuit_todd(todd_path, circuit_qiskit):
    """
    Optimized a Qiskit QuantumCircuit with TODD.
    
    Args:
        todd_path (str): Path to TODD installation
        circuit_qiskit (qiskit.QuantumCircuit): quantum circuit input
    
    Returns:
        opt_circ: optimized quantum circuit in qiskit format
    """
    n_ancillas = 0
    out_path = f'{os.path.dirname(todd_path)}/circuit'
    os.makedirs(out_path, exist_ok=True)

    # convert qasm to custom tfc format used by TODD
    tfc_str = qasm_to_tfc(qiskit.qasm2.dumps(circuit_qiskit))

    # save file
    with open(f'{out_path}/circuit.tfc', 'w') as f:
        f.write(tfc_str)

    # optimize with TODD
    ## >/dev/null 2>&1 to supress output
    os.system(f'{todd_path} circuit {out_path}/circuit.tfc -a todd -h {n_ancillas} -o {out_path}/circ.opt >/dev/null 2>&1')

    # load result of TODD
    with open(f'{out_path}/circ.opt', 'r') as f:
        opt_circ = qiskit.qasm2.loads(qasm_from_todd_output(f.read()))
    
    return opt_circ

def process_circuit(circuit_qiskit):
    """
    Process a Qiskit QuantumCircuit following step 6 of appendix C.1 of the AlphaTensor-Quantum paper.
    In this case, U1 and U3 does not exist, we only need to process U2.
    The output of this function is the circuit W.
    
    Args:
        circuit_qiskit (qiskit.QuantumCircuit): quantum circuit input which is U2
    
    Returns:
        circ: The W circuit that contains only CNOT and T gates to be processed by AlphaTensor-Quantum
    """
    # Get U2, U2 contains only CNOT and phase gate
    U2_circ = circuit_qiskit

    # Let C be the circuit formed by taking just the CNOT gates in U2
    C_circ = get_cnot_circuit(U2_circ)

    # Get U2' = C^{-1}U2
    U2_prime_circ = U2_circ.compose(C_circ.inverse())

    # W be the circuit formed by removing all S and Z gates from U2', W contains only CNOT and T gates
    W_circ = get_cnot_t_circuit(U2_prime_circ)

    return W_circ 

def count_t_gate(circuit_qiskit):
    """
    Count the number of T gate in a Qiskit QuantumCircuit.
    
    Args:
        circuit_qiskit (qiskit.QuantumCircuit): quantum circuit input
    
    Returns:
        tcount: the number of T gate 
    """ 
    t_countdict = dict(circuit_qiskit.count_ops())
    if 't' in t_countdict.keys():
        tcount = t_countdict['t']
    else:
        tcount = 0

    return tcount
 
# Convert to the list format
def circuit_to_gate_list(circuit_qiskit):
    """
    Convert Qiskit QuantumCircuit into a gate list format for AlphaTensor Quantum.
    
    Args:
        circuit_qiskit (qiskit.QuantumCircuit): quantum circuit input
    
    Returns:
        gate_list: list of gates of the circuits
    """ 
    gate_list = []
    for gate, qargs, _ in circuit_qiskit.data:
        if gate.name == 't':
            gate_list.append(f"T({qargs[0]._index})")
        elif gate.name == 'cx':
            gate_list.append(f"CX({qargs[0]._index}, {qargs[1]._index})")
    return gate_list

def get_cnot_circuit(original_circuit):
    """
    Process a Qiskit QuantumCircuit to include only CX gates.
    
    Args:
        circuit_qiskit (qiskit.QuantumCircuit): quantum circuit input
    
    Returns:
        new_circ: quantum circuit in qiskit format
    """
    # Create a new circuit with the same number of qubits
    cnot_only_circuit = QuantumCircuit(original_circuit.num_qubits)

    # Iterate through the instructions of the original circuit
    for instruction in original_circuit.data:
        if instruction.operation.name == "cx":  # Check if the gate is a CNOT
            qubits = [qubit._index for qubit in instruction.qubits]
            cnot_only_circuit.cx(qubits[0], qubits[1])
            
    return cnot_only_circuit

def get_cnot_t_circuit(original_circuit):
    """
    Process a Qiskit QuantumCircuit to include only T and CX gates.
    
    Args:
        circuit_qiskit (qiskit.QuantumCircuit): quantum circuit input
    
    Returns:
        new_circ: quantum circuit in qiskit format
    """
    # Create a new circuit with the same number of qubits
    cnot_only_circuit = QuantumCircuit(original_circuit.num_qubits)

    # Iterate through the instructions of the original circuit
    for instruction in original_circuit.data:
        if instruction.operation.name == "cx":  # Check if the gate is a CNOT
            qubits = [qubit._index for qubit in instruction.qubits]
            cnot_only_circuit.cx(qubits[0], qubits[1])
        if instruction.operation.name == "t":  # Check if the gate is a CNOT
            qubits = [qubit._index for qubit in instruction.qubits]
            cnot_only_circuit.t(qubits[0])
            
    return cnot_only_circuit

def get_cnot_sz_circuit(original_circuit):
    """
    Process a Qiskit QuantumCircuit to include only S, Z, and CX gates.
    
    Args:
        circuit_qiskit (qiskit.QuantumCircuit): quantum circuit input
    
    Returns:
        new_circ: quantum circuit in qiskit format
    """
    # Create a new circuit with the same number of qubits
    cnot_only_circuit = QuantumCircuit(original_circuit.num_qubits)

    # Iterate through the instructions of the original circuit
    for instruction in original_circuit.data:
        if instruction.operation.name == "cx":  # Check if the gate is a CNOT
            qubits = [qubit._index for qubit in instruction.qubits]
            cnot_only_circuit.cx(qubits[0], qubits[1])
        elif instruction.operation.name == "s":  # Check if the gate is a CNOT
            qubits = [qubit._index for qubit in instruction.qubits]
            cnot_only_circuit.s(qubits[0])
        elif instruction.operation.name == "z":  # Check if the gate is a CNOT
            qubits = [qubit._index for qubit in instruction.qubits]
            cnot_only_circuit.z(qubits[0])
            
    return cnot_only_circuit
