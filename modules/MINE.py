import torch.nn as nn
from torch.nn import functional as F

class LinearMine(nn.Module):
    def __init__(self, code_qubits, output_qubits, size=50):
        super(LinearMine, self).__init__()
        self.fc1 = nn.Linear(code_qubits, size)
        self.fc2 = nn.Linear(output_qubits, size)
        self.fc3 = nn.Linear(size, 1)

    def forward(self, x, y):
        h1 = F.relu(self.fc1(x)+self.fc2(y))
        h2 = self.fc3(h1)
        return h2