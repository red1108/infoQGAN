import pennylane as qml
import torch
import numpy as np

class QGenerator:
    def __init__(self, n_qubits, output_qubits, n_layers, params, dev, entangling="CNOT"):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.params = params

        self.output_qubits = output_qubits

        if output_qubits > n_qubits:
            raise ValueError("output_qubits should be smaller than n_qubits")
        
        self.dev = dev  # pennylane device
        self.generator_circuit_qnode = qml.QNode(self.circuit, self.dev, interface="torch")
        self.entangling = entangling
        assert entangling in ["CNOT", "CZ"], "entangling should be 'CNOT' or 'CZ'"
        
    def init_circuit(self, generator_input):
        for i in range(self.n_qubits):
            qml.RY(generator_input[i]*np.pi/2, wires=i)
        
    def single_layer(self, params, last=False):
        for i in range(self.n_qubits):
            qml.RY(params[i][0], wires=i)
        
        if not last:
            for i in range(self.n_qubits):
                if self.entangling == "CNOT":
                    qml.CNOT(wires=[i, (i+1)%self.n_qubits])
                elif self.entangling == "CZ":
                    qml.CZ(wires=[i, (i+1)%self.n_qubits])

    def circuit(self, generator_input):
        # output dimension: 2**output_qubits

        self.init_circuit(generator_input)

        for i in range(self.n_layers):
            self.single_layer(self.params[i], last=(i==self.n_layers-1))

        return qml.probs(wires=range(self.output_qubits)) # |00>, |01>, |10>, |11> 이런식으로 모든 basis들의 확률값을 반환

    def forward(self, generator_input):
        generator_output = [self.generator_circuit_qnode(single_in) for single_in in generator_input]  # (BATCH_SIZE, 2**output_qubits)
        generator_output = torch.stack(generator_output)  # (BATCH_SIZE, 2**output_qubits)
        return generator_output
    
    def parameters(self):
        return [self.params]
    
    def state_dict(self):
        return self.params
