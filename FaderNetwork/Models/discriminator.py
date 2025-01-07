import torch.nn as nn
        
class Discriminator(nn.Module):
    def __init__(self, latent_channels=512, hidden = 512, n_attr = 40, attributes=[]):
        """
        Initialise les paramètres et les couches du discriminateur.
        
        Arguments :
        - latent_channels : Nombre de canaux en entrée (taille des caractéristiques latentes).
        - hidden : Taille de la couche dense cachée.
        - n_attr : Nombre total d'attributs à prédire.
        - attributes : Liste des noms des attributs (facultatif, utile pour l'interprétation).
        """
        self.attributes = attributes
        self.n_attr = n_attr
        super(Discriminator, self).__init__()
        self.latent = nn.Sequential(
            # 1. Première couche : Convolution 2D
            nn.Conv2d(latent_channels, 512, kernel_size=4, stride=2, padding=1),   
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 2. Aplatissement de la sortie convolutive
            nn.Flatten(),
            # 3. Couche dense 1 : Réduction vers une taille intermédiaire
            nn.Linear(latent_channels, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            # 4. Couche dense 2 : Prédiction finale
            nn.Linear(hidden, 2*n_attr),
        )

    def forward(self, x):
        """
        Effectue une propagation avant à travers le discriminateur.
        
        Paramètres :
        - x : Tenseur d'entrée représentant le vecteur latent ou une image.
        
        Retourne :
        - Un vecteur de taille [batch_size, 2 * n_attr] représentant les scores pour chaque attribut.
        """
        return self.latent(x)
