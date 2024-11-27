
# NODIER David code Decoder

# On import les librairies
import torch
import torch.nn as nn

# On crée la classe Décoder

class Decoder(nn.Module) :
    def __init__(self, dim = 512, attribut = 10) :

        super(Decoder, self).__init__()
        
        # On code l'architecture en suivant les paramètres pris par l'article comme la taille du kernel, du stride...
        # Architecture symétrique au Fader Encoder et on utilise une entrée de dim + 2n avec n le nbre d'attribut
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim + 2 * attribut, 512 + 2 * attribut, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(512 + 2 * attribut, 256 + 2 * attribut, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256 + 2 * attribut, 128 + 2 * attribut, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128 + 2 * attribut, 64 + 2 * attribut, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64 + 2 * attribut, 32 + 2 * attribut, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32 + 2 * attribut, 16 + 2 * attribut, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(16 + 2 * attribut, 3 + 2 * attribut, kernel_size=4, stride=2, padding=1),  # Dernière couche
            nn.Tanh()  # Normalisation de la sortie dans [-1, 1]
        )

    # On definition le fonction forward qui définit le déroulé dans le décodeur  
    def forward(self, latent_code, attributes):
       
        # On concaténe les attributs codés en one-hot au latent_code
        # Le but est d'avoir un vecteur [1, 0] ou [0, 1]
        attributes_onehot = torch.cat([attributes, 1 - attributes], dim=1) 
        
        # On rajoute des dimensions spatiales comme la hauteur et la largeur representé par (dim = 2 et 3) 
        attributes_onehot = attributes_onehot.unsqueeze(2).unsqueeze(3)  

        # On fait "expand" pour faire correspondre les dimensions aux dimensions du latent code (qui est la sortie du encodeur)
        attributes_onehot = attributes_onehot.expand(-1, -1, latent_code.size(2), latent_code.size(3))
        
        # On ajoute les attributes_onehot qu'on a adapté au code latent (sortie de l'encodeur)
        decoder_input = torch.cat([latent_code, attributes_onehot], dim=1)  
        
        # On passe dans le décodeur
        output = self.decoder(decoder_input)
        return output

# Provisoire
if __name__ == "__main__":
    # Paramètres
    latent_dim = 512
    num_attributes = 10
    batch_size = 32
    image_size = 256

    # Input provisoire
    latent_code = torch.randn(batch_size, latent_dim, 2, 2)  # Taille latente
    attributes = torch.randint(0, 2, (batch_size, num_attributes))  # Attributs binaires

    # Initialisation du décodeur
    decoder = Decoder(latent_dim=latent_dim, num_attributes=num_attributes)

    # Génération de l'image
    reconstructed_images = decoder(latent_code, attributes)
    print(f"Taille des images reconstruites : {reconstructed_images.shape}")  
