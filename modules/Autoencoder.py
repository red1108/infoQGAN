import torch
import torch.nn as nn
class NormalizeLayer(nn.Module):
    def forward(self, x):
        return x / x.sum(dim=1, keepdim=True)
class AbsoluteValueLayer(nn.Module):
    def forward(self, x):
        return torch.abs(x)
class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 28 * 28, latent_dim),  # 입력 크기를 8 * 28 * 28로 수정
            AbsoluteValueLayer(),  # 양수로 만드는 레이어 추가
            
            NormalizeLayer()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8 * 28 * 28),  # 출력 크기 수정
            nn.ReLU(),
            nn.Unflatten(1, (8, 28, 28)),
            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
