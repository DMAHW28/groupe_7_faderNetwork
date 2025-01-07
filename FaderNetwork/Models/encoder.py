import torch.nn as nn


#l'encodeur prend une image couleur
#en respectant l'architecture décrite dans l'article nous avons
class Encoder(nn.Module):
    def __init__(self, latent_dim=512, input_channel = 3):
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            #première couche C16, la dimension de l'image est réduite de 256 -> 128
            nn.Conv2d(input_channel, 16, kernel_size=4, stride=2, padding=1),  # 256x256 -> 128x128
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            
            #deuxième couche C32, la dimension de l'image est réduite de 128 -> 64
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # 128x128 -> 64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # C64 64 -> 32
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # C128 32 -> 16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # C256 16 -> 8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # C512 8 -> 4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 8x8 -> 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # C512 4 -> 2
            nn.Conv2d(512, latent_dim, kernel_size=4, stride=2, padding=1),  # 4x4 -> 2x2
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        """
        Passe une image à travers l'encodeur pour obtenir la représentation latente.
        
        En entrée :
            x (torch.Tensor): Tensor d'image de forme (batch_size, 3, 256, 256).
        
        Et retourne:
            torch.Tensor: Représentation latente de forme (batch_size, latent_dim, 2, 2).
        """
        latent = self.conv_layers(x)
        return latent
