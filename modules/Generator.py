import torch.nn as nn
import torch
class LinearGeneratorSigmoid(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        return self.layers(z)