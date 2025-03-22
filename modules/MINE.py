import torch.nn as nn
from torch.nn import functional as F

class LinearMine(nn.Module):
    def __init__(self, code_dim, output_dim, size=50):
        super(LinearMine, self).__init__()
        self.fc1 = nn.Linear(code_dim, size)
        self.fc2 = nn.Linear(output_dim, size)
        self.fc3 = nn.Linear(size, 1)

    def forward(self, x, y):
        h1 = F.relu(self.fc1(x)+self.fc2(y))
        h2 = self.fc3(h1)
        return h2