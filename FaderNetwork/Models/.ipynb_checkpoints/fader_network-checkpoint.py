import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder

class FaderNetwork(nn.Module):
    def __init__(self, input_channels=3, latent_channels=512, attribute_dim=40):
        super(FaderNetwork, self).__init__()
        self.encoder = Encoder(input_channels, latent_channels)
        self.decoder = Decoder(latent_channels, attribute_dim)
        # self.decoder = Decoder(latent_channels, attribute_dim, input_channels)
    
    def forward(self, x, attributes):
        latent = self.encoder(x)  # [batch_size, 512, 2, 2]
        reconstructed = self.decoder(latent, attributes)  # [batch_size, 3, 256, 256]
        return latent, reconstructed
