import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder

class FaderNetwork(nn.Module):
    def __init__(self, input_channels=3, latent_channels=512, attribute_dim=40, attributes=[]):
        super(FaderNetwork, self).__init__()
        self.attributes = attributes
        self.encoder = Encoder(latent_channels, input_channels)
        self.decoder = Decoder(latent_channels, attribute_dim)
    
    def forward(self, x, attributes):
        latent = self.encoder(x)  # [batch_size, 512, 2, 2]
        reconstructed = self.decoder(latent, attributes)  # [batch_size, 3, 256, 256]
        return latent, reconstructed
