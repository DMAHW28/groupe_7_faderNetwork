import torch.nn as nn

class Latent(nn.Module):
    def __init__(self, latent_channels=512, hidden = 512, n_attr = 40):
        self.n_attr = n_attr
        super(Latent, self).__init__()
        self.latent = nn.Sequential(
            #première couche C16, la dimension de l'image est réduite de 256 -> 128
            nn.Conv2d(latent_channels, 512, kernel_size=4, stride=2, padding=1),   
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(latent_channels, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 2*n_attr),
            # nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.latent(x).view(-1, self.n_attr, 2)

    
class Classifier(nn.Module):
    def __init__(self, input_channels=3, hidden = 512, n_attr = 40):
        super(Classifier, self).__init__()
        self.n_attr = n_attr
        self.classifier = nn.Sequential(
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
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(2*2*512, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 2*n_attr),   
            # nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.classifier(x).view(-1, self.n_attr, 2)

    
class Patch(nn.Module):
    def __init__(self, input_channels=3, hidden = 512):
        super(Patch, self).__init__()
        self.patch = nn.Sequential(
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
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(2*2*512, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 1),   
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.patch(x)
    