import torch.nn as nn

# 오토인코더 정의
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=40):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 7 * 7, latent_dim),
            nn.Tanh()  # -1 ~ 1로 출력 제한
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (8, 7, 7)),
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # 복원된 이미지를 0~1로 제한
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        latent = (latent + 1) / 2  # -1 ~ 1을 0 ~ 1로 스케일링
        reconstructed = self.decoder(latent)
        return reconstructed
