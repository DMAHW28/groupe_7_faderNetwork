
# NODIER David code Decoder

# On import les librairies
import torch
import torch.nn as nn

# On crée la classe Décoder

"""class Decoder(nn.Module) :
    def __init__(self, dim = 512, attribut = 10) :

        super(Decoder, self).__init__()
        
        # On code l'architecture en suivant les paramètres pris par l'article comme la taille du kernel, du stride...
        # Architecture symétrique au Fader Encoder et on utilise une entrée de dim + 2n avec n le nbre d'attribut
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim + 2 * attribut, 512 + 2 * attribut, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512 + 2 * attribut),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(512 + 2 * attribut, 256 + 2 * attribut, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256 + 2 * attribut),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256 + 2 * attribut, 128 + 2 * attribut, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128 + 2 * attribut),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128 + 2 * attribut, 64 + 2 * attribut, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64 + 2 * attribut),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64 + 2 * attribut, 32 + 2 * attribut, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32 + 2 * attribut),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32 + 2 * attribut, 16 + 2 * attribut, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16 + 2 * attribut),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(16 + 2 * attribut, 3, kernel_size=4, stride=2, padding=1),  # Dernière couche
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
        return output"""

# On crée la classe Décoder
class Decoder(nn.Module):
    def __init__(self, latent_dim=512, num_attributes=3):
        super(Decoder, self).__init__()

        # On définit les dimensions utilisées dans le décodeur
        # latent_dim représente la taille de l'espace latent
        # num_attributes est le nombre d'attributs binaires
        # attribute_dim correspond à 2 * num_attributes car chaque attribut est codé en one-hot
        # input_dim, il s'agit de la dimension d'entrée pour la première couche du décodeur
        self.num_attributes = num_attributes
        self.attribute_dim = 2*num_attributes
        self.input_dim = latent_dim + self.attribute_dim

        # On code l'architecture en suivant les paramètres pris par l'article comme la taille du kernel, du stride...
        # Chaque couche reçoit les attributs concaténés avec les features issues de la couche précédente
        self.dec1 = self.layer_decoder(canal=self.input_dim , filter_size=512)
        self.dec2 = self.layer_decoder(canal=512 + self.attribute_dim, filter_size=512)
        self.dec3 = self.layer_decoder(canal=512 + self.attribute_dim, filter_size=256)
        self.dec4 = self.layer_decoder(canal=256 + self.attribute_dim, filter_size=128)
        self.dec5 = self.layer_decoder(canal=128 + self.attribute_dim , filter_size=64)
        self.dec6 = self.layer_decoder(canal=64 + self.attribute_dim, filter_size=32)
        self.dec7 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(32 + self.attribute_dim, 3, kernel_size=4, stride=2, padding=1),
            torch.nn.Tanh()
        )
        
    def forward(self, latent, attributes):
        batch_size = latent.size(0)
        # On rajoute des dimensions spatiales comme la hauteur et la largeur representé par (dim = 2 et 3) 
        attributes = attributes.unsqueeze(2).unsqueeze(3)
        input_1 = torch.cat([latent, attributes.expand(batch_size, self.attribute_dim, latent.size(2), latent.size(2))], 1)

        dec1 = self.dec1(input_1)
        input_2 = torch.cat([dec1, attributes.expand(batch_size, self.attribute_dim, dec1.size(2), dec1.size(2))], 1)
        
        dec2 = self.dec2(input_2)
        input_3 = torch.cat([dec2, attributes.expand(batch_size, self.attribute_dim, dec2.size(2), dec2.size(2))], 1)
        
        dec3 = self.dec3(input_3)
        input_4 = torch.cat([dec3, attributes.expand(batch_size, self.attribute_dim, dec3.size(2), dec3.size(2))], 1)
        
        dec4 = self.dec4(input_4)
        input_5 = torch.cat([dec4, attributes.expand(batch_size, self.attribute_dim, dec4.size(2), dec4.size(2))], 1)
        
        dec5 = self.dec5(input_5)
        input_6 = torch.cat([dec5, attributes.expand(batch_size, self.attribute_dim, dec5.size(2), dec5.size(2))], 1)
        
        dec6 = self.dec6(input_6)
        input_7 = torch.cat([dec6, attributes.expand(batch_size, self.attribute_dim, dec6.size(2), dec6.size(2))], 1)
        output = self.dec7(input_7)
        return output
    
    def layer_decoder(self, canal, filter_size):
        return torch.nn.Sequential(
            torch.nn.ConvTranspose2d(canal, filter_size, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(filter_size),
            torch.nn.ReLU(inplace=True)
        )



