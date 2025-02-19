import re

qasm_to_tfc_gate = {'rz(pi/4)': 'T',
                    'rz(-pi/4)': 'T\'',
                    'rz(pi/2)': 'S',
                    'rz(-pi/2)': 'S\'',
                    'x': 'X',
                    'y': 'Y',
                    'z': 'Z',
                    'h': 'H',
                    's': 'S',
                    'sdg': 'S\'',
                    't': 'T',
                    'tdg': 'T\'',
                    'cx': 'CNOT',
                    'ccx': 'Toffoli',
                    'cz': 'CZ',
                    'ccz': 'CCZ',
                    'id': 'I',
                    'project0': 'M',}

tfc_gate_to_qasm = {v: k for k, v in qasm_to_tfc_gate.items()}

def qasm_to_tfc(qasm):
    tfc_str = '.v'
    for i, line in enumerate(qasm.splitlines()):
        if line.startswith('qreg'):
            initial_line = i
            break
    qasm = qasm.split("\n")[initial_line:]

    # Top line defining qubit names
    qubits = re.findall(r'\d+', qasm[0])
    assert len(qubits) == 1, 'only one qubit register supported'
    n_qubits = int(qubits[0])

    for i in range(0, n_qubits):
        tfc_str += f' {i}'
    tfc_str += '\nBEGIN\n'

    for gate_str in qasm[1:]:
        split = gate_str.split(" ")
        gate = split[0]
        targets = [int(x) for x in re.findall(r'\d+', split[1])]#[::-1]
        tfc_gate = qasm_to_tfc_gate[gate]
        tfc_str += tfc_gate
        for target in targets:
            tfc_str += f' {target}'
        tfc_str += '\n'
    tfc_str += 'END\n'
    
    return tfc_str

def tfc_to_qasm(tfc):
    '''this function is limited to gate sin '''
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'

    tfc = tfc.split("\n")

    assert tfc[0].startswith('.v'), 'First line must start with .v'
    assert tfc[1] == 'BEGIN', 'Second line must be BEGIN'
    while tfc[-1] == '':
        tfc = tfc[:-1]
    assert tfc[-1] == 'END', 'Last line must be END'

    qubits = re.findall(r'\d+', tfc[0])
    n_qubits = len(qubits)
    qasm += f'qreg q[{n_qubits}];\n'

    for line in tfc[2:-1]:
        gate = tfc_gate_to_qasm[line.split(" ")[0]]
        qasm += f'{gate} '
        targets = [int(x) for x in re.findall(r'\d+', line)]#[::-1]
        for target in targets:
            qasm += f'q[{target}],'
        qasm += ';\n'

    return qasm

def qasm_from_todd_output(todd_out_str):
    split = todd_out_str.split("\n")
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
    i = 0
    for line in split:
        if line.startswith('Output circuit:'):
            break
        i += 1
    n_qubits = int(re.findall(r'\d+', split[i+1])[0])
    qasm += f'qreg q[{n_qubits}];\n'
    for line in split[i+6:]:
        if line == '':
            break

        gate = tfc_gate_to_qasm[line.split(" ")[0]]
        qasm += f'{gate} '

        targets = [int(x) for x in re.findall(r'\d+', line)][::-1]
        for target in targets:
            qasm += f'q[{target-1}],'
        qasm += ';\n'

    return qasm


