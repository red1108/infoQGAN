import torch.nn as nn
import torch

from modules.Autoencoder import AbsoluteValueLayer, NormalizeLayer

from abc import ABC, abstractmethod

# Generator
class LinearGeneratorDirichlet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=50):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
            AbsoluteValueLayer(),  # 총 합으로 나누어 정규화하기 전에, 모든 값을 절댓값 씌워서 양수로 만듦.
            NormalizeLayer() # 총 합으로 나누어서 합이 1이 되도록 정규화.
        )
    
    def forward(self, z):
        return self.layers(z)

class LinearGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=50):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )
    
    def forward(self, z):
        return self.layers(z)


class LinearGeneratorSigmoid(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=50):
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