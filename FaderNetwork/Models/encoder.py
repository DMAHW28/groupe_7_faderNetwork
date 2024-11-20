import torch.nn as nn


#l'encodeur prend une image couleur
#en respectant l'architecture décrite dans l'article nous avons
class Encoder(nn.Module):
    def __init__(self, input_channels=3, latent_channels=512):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            #première couche C16, la dimension de l'image est réduite de 256 -> 128
            nn.Conv2d(input_channels, 16, kernel_size=4, stride=2, padding=1),   
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            #deuxième couche C32, la dimension de l'image est réduite de 128 -> 64
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # C64 64 -> 32
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # C128 32 -> 16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # C256 16 -> 8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # C512 8 -> 4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # C512 4 -> 2
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        latent = self.encoder(x) 
        return latent
