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
    
class CNNDiscriminator(nn.Module):
    def __init__(self, img_size=28):
        super(CNNDiscriminator, self).__init__()
        self.img_size = img_size

        # stride=1로 변경
        self.conv_layers = nn.Sequential(
            # Conv1: 채널 1 -> 16, kernel_size=4, stride=1, padding=1
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Conv2: 채널 16 -> 32, kernel_size=4, stride=1, padding=1
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(16 * (img_size - 2) * (img_size - 2), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (batch_size, H, W),  예: H=W=img_size
        """
        # 채널 차원 추가: (B, 1, H, W)
        if x.ndim == 3:
            x = x.unsqueeze(1)

        # 합성곱 통과
        x = self.conv_layers(x)
        # (B, out_channels, out_H, out_W) -> (B, out_channels*out_H*out_W)
        x = x.view(x.size(0), -1)

        # 최종 출력 (B, 1)
        out = self.fc(x)
        return out

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