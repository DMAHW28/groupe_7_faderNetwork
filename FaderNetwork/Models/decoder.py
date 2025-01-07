
# NODIER David code Decoder

# On import les librairies
import torch
import torch.nn as nn

# On crée la classe Décoder
class Decoder(nn.Module):
    def __init__(self, latent_dim=512, num_attributes=3):
        super(Decoder, self).__init__()

        # On définit les dimensions utilisées dans le décodeur
        # latent_dim représente la taille de l'espace latent
        # num_attributes est le nombre d'attributs binaires
        # attribute_dim correspond à 2 * num_attributes 
        # input_dim, il s'agit de la dimension d'entrée pour la première couche du décodeur
        self.num_attributes = num_attributes
        self.attribute_dim = 2*num_attributes
        self.input_dim = latent_dim + self.attribute_dim

        # On code l'architecture en suivant les paramètres pris par l'article comme la taille du kernel, du stride...
        # Chaque couche reçoit les attributs concaténés avec les features issues de la couche précédente
        # Comme il est mentionné dans l'article, le nombre de canaux en entrée commence à 512 + 2n
        self.dec1 = self.layer_decoder(canal=self.input_dim , filter_size=512)
        self.dec2 = self.layer_decoder(canal=512 + self.attribute_dim, filter_size=512)
        self.dec3 = self.layer_decoder(canal=512 + self.attribute_dim, filter_size=256)
        self.dec4 = self.layer_decoder(canal=256 + self.attribute_dim, filter_size=128)
        self.dec5 = self.layer_decoder(canal=128 + self.attribute_dim , filter_size=64)
        self.dec6 = self.layer_decoder(canal=64 + self.attribute_dim, filter_size=32)

        # La dernière couche utilise une activation Tanh pour normaliser les sorties entre -1 et 1
        self.dec7 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(32 + self.attribute_dim, 3, kernel_size=4, stride=2, padding=1),
            torch.nn.Tanh()
        )
        
    def forward(self, latent, attributes):

        # On récupère la taille du batch
        batch_size = latent.size(0)
        
        # On ajuste la dimension des attributs (comme la hauteur et la largeur) pour qu'ils puissent être concaténés
        attributes = attributes.unsqueeze(2).unsqueeze(3)

        # On concatène le latent avec les attributs
        # On utilise "expand" pour faire correspondre les dimensions des attributs aux dimensions du latent code (qui est la sortie de l'encodeur)
        input_1 = torch.cat([latent, attributes.expand(batch_size, self.attribute_dim, latent.size(2), latent.size(2))], 1)

        # On réalise des concaténations en passant par chaque couche du décodeur 
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

        # On génère la sortie finale avec la dernière couche
        output = self.dec7(input_7)
        return output
    
    def layer_decoder(self, canal, filter_size):
        # Dans notre cas, chaque couche du décoder est composé de 3 éléments :convolution, normalisation et fonction d'activation ReLU
        return torch.nn.Sequential(
            torch.nn.ConvTranspose2d(canal, filter_size, kernel_size=4, stride=2, padding=1), # Convolution
            torch.nn.BatchNorm2d(filter_size), # Normalisation
            torch.nn.ReLU(inplace=True) #Fonction d'activation ReLU
        )



