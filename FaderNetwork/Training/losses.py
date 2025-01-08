import torch.nn as nn
import torch

def attr_loss(output, attributes, flip):
    """
    Calcule la perte pour les attributs catégoriels du discriminateur.
    Args:
        output (torch.Tensor): Les prédictions du modèle de taille (batch_size, 2*n_attributes).
        attributes (torch.Tensor): Les attributs cibles (labels) de taille (batch_size, 2*n_attributes).
                                   Chaque attribut est encodé sur deux colonnes [0, 1] pour la valeur 1, [1, 0] pour la valeur 0.
        flip (bool): Si True, les labels cibles sont aléatoirement modifiés.
    Returns:
        torch.Tensor: La perte totale sur tous les attributs pour l'ensemble des prédictions du batch.
    """
    loss = 0 # Initialisation de la perte totale
    n_cat = 2 # Chaque attribut est encodé sur deux colonnes (codage one hot)
    for k in range(0, attributes.size(1), n_cat):
        # Sélectionner les prédictions pour l'attribut courant
        x = output[:, k:k + n_cat].contiguous()
        # Obtenir les labels cibles pour cet attribut
        y = attributes[:, k:k + n_cat].argmax(1)
        if flip: # Si flip est activé, on modifie aléatoirement les labels
            y = (y + torch.randint(1, n_cat, y.size(), device=y.device)) % n_cat
        # Calcul de la perte pour cet attribut en utilisant l'entropie croisée
        loss += nn.functional.cross_entropy(x, y)
    return loss
