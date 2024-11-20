import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, num_attributes):
        super(Discriminator, self).__init__()
        self.num_attributes = num_attributes
        
        # Définition des couches
        self.conv = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)  # C512
        self.fc1 = nn.Linear(512 * 2 * 2, 512)  # Fully connected layer
        self.fc2 = nn.Linear(512, num_attributes)  # Sortie pour prédire les attributs
        
        # Activation et régularisation
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)  # Régularisation Dropout

    def forward(self, z):
        x = self.leaky_relu(self.conv(z))  # Convolution
        x = x.view(x.size(0), -1)  # Flatten
        x = self.leaky_relu(self.fc1(x))  # Fully connected 1
        x = self.dropout(x)  # Appliquer Dropout
        x = self.fc2(x)  # Fully connected 2 (sortie)
        return x
