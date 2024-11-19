
# NODIER David code Decoder

# On import les librairies
import torch
import torch.nn as nn

# On crée la classe Décoder

class Decoder(nn.Module) :
    def __init__(self, dim = 512, attribut = 10) :

        super(Decoder, self).__init__()

        # Taille entrée
        self.taille_entree = dim + 2 * attribut

        # On code l'architecture
        # Architecture symétrique au Fader Encoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.input_dim, 512, kernel_size=4, stride=2, padding=1),  # Up-sampling
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),  # Dernière couche : 3 canaux (RGB)
            nn.Tanh()  # Normalisation de la sortie dans [-1, 1]
        )


    def forward(self, latent_code, attributes):
       
        # Concaténer les attributs codés en one-hot au latent_code
        attributes_onehot = torch.cat([attributes, 1 - attributes], dim=1)  # Taille : (batch_size, 2*num_attributes)
        attributes_onehot = attributes_onehot.unsqueeze(2).unsqueeze(3)  # Ajouter des dimensions spatiales
        attributes_onehot = attributes_onehot.expand(-1, -1, latent_code.size(2), latent_code.size(3))  # Diffusion
        
        # Ajouter les attributs au code latent
        decoder_input = torch.cat([latent_code, attributes_onehot], dim=1)  # Taille : (batch_size, 512 + 2*num_attributes, H, W)
        
        # Passer dans le décodeur
        output = self.decoder(decoder_input)
        return output

# Exemple d'utilisation
if __name__ == "__main__":
    # Paramètres
    latent_dim = 512
    num_attributes = 10
    batch_size = 8
    image_size = 256

    # Dummy inputs
    latent_code = torch.randn(batch_size, latent_dim, 2, 2)  # Taille latente
    attributes = torch.randint(0, 2, (batch_size, num_attributes))  # Attributs binaires

    # Initialisation du décodeur
    decoder = FaderDecoder(latent_dim=latent_dim, num_attributes=num_attributes)

    # Génération de l'image
    reconstructed_images = decoder(latent_code, attributes)
    print(f"Taille des images reconstruites : {reconstructed_images.shape}")  # (batch_size, 3, 256, 256)
