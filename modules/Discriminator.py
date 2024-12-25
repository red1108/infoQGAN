import torch.nn as nn
from torch.nn import functional as F

class LinearDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_size=50):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid())

        self.input_dim = input_dim

    def forward(self, x):
        if(len(x.shape) != 2):
            x = x.view(x.shape[0], -1)

        return self.layers(x)
    


class QDiscriminator():
    def __init__(self, n_qubits, n_layers, params, dev):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.params = params

        self.dev = dev  # pennylane device
        self.generator_circuit_qnode = qml.QNode(self.circuit, self.dev, interface="torch")
        
    def init_circuit(self, initial_state):
        qml.StatePrep(initial_state, pad_with=0., validate_norm=True, wires=range(self.n_qubits))
        
    def single_layer(self, params):
        for i in range(self.n_qubits):
            qml.RX(params[i][0], wires=i)
            qml.RY(params[i][1], wires=i)
            qml.RZ(params[i][2], wires=i)
        
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i+1)%self.n_qubits])

    def circuit(self, generator_input):
        # output dimension: 2**output_qubits

        self.init_circuit(generator_input)

        for param in self.params:
            self.single_layer(param)

        return qml.probs(wires=0) # 첫 큐빗이 [0일 확률, 1일 확률]

    def forward(self, initial_states):
        generator_output = [self.generator_circuit_qnode(initial_state)[0] for initial_state in initial_states]  # (BATCH_SIZE, 1)
        generator_output = torch.stack(generator_output)  # (BATCH_SIZE, 1)
        return generator_output
    
    def parameters(self):
        return [self.params]