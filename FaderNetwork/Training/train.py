LAMBDA_E_max = 1e-4 # Valeur maximale de lambda_e pour le poids de la perte adversaire

def accuracy_latent(output, attributes):
    """
    Calcule la précision pour les prédictions du discriminateur latent.
    Args:
        output (torch.Tensor): Les prédictions du modèle (batch_size, 2*n_attributes).
        attributes (torch.Tensor): Les attributs cibles (batch_size, 2*n_attributes).
    Returns:
        float: Précision totale sur le batch.
    """
    accuracy = 0 # Initialisation de la précision totale
    n_cat = 2 # Chaque attribut est encodé sur deux colonnes (codage one hot)
    for k in range(0, attributes.size(1), n_cat):
        # Sélectionner les prédictions pour l'attribut courant
        pred = output[:, k:k + n_cat].argmax(1)
        # Obtenir les labels cibles pour cet attribut
        y = attributes[:, k:k + n_cat].argmax(1)
        # Vérification des correspondances entre prédictions et cibles
        accuracy += pred.eq(y.view_as(pred)).sum().item()
    return accuracy

class Trainer:
    """
    Classe pour l'entraînement de l'autoencodeur (AE) et le discriminateur.
    """
    def __init__(self, ae, latent, it_max):
        self.ae = ae # Autoencodeur
        self.latent = latent # discriminateur
        self.latent_train_loss = 0 # Perte totale pour le discriminateur
        self.ae_train_loss = 0 # Perte totale pour l'Autoencodeur
        self.lambda_e = 0 # Poids pour la perte adversaire
        self.iterations = 0  # Compteur des itérations
        self.total_it = it_max # Nombre d'itérations maximal
        self.adv_loss = 0 # Perte adversaire

    def init_parameters(self):
        """Réinitialise les paramètres d'entraînement."""
        self.latent_train_loss = 0
        self.ae_train_loss = 0
        self.adv_loss = 0

    def latent_step(self, X, y, criterion, optim):
        """
        Effectue une étape d'entraînement pour le discriminateur.
        Args:
            X (torch.Tensor): Données d'entrée.
            y (torch.Tensor): Labels d'attributs.
            criterion (callable): Fonction de perte.
            optim (torch.optim.Optimizer): Optimiseur pour le modèle discriminateur.
        """
        self.ae.eval()
        self.latent.train()
        optim.zero_grad()
        z = self.ae.encoder(X) # Encoder les données pour obtenir les représentations latentes
        y_pred = self.latent(z) # Prédire les attributs avec le modèle latent
        loss = criterion(y_pred, y, False) # Calculer la perte
        self.latent_train_loss += loss.item() # Accumuler la perte
        loss.backward() # Propagation arrière
        optim.step() # Mise à jour des poids

    def ae_step(self, X, y, criterion_ae, criterion_latent, optim):
        """
        Effectue une étape d'entraînement pour l'autoencodeur.
        Args:
            X (torch.Tensor): Données d'entrée.
            y (torch.Tensor): Labels d'attributs.
            criterion_ae (callable): Fonction de perte pour l'autoencodeur.
            criterion_latent (callable): Fonction de perte pour le discriminateur.
            optim (torch.optim.Optimizer): Optimiseur pour l'autoencodeur.
        """
        self.latent.eval()
        self.ae.train()
        optim.zero_grad()
        z, D = self.ae(X, y) # Passer les données à travers l'autoencodeur
        y_pred = self.latent(z) # Prédire les attributs à partir de la représentation latente
        loss_adv = criterion_latent(y_pred, y, True) # Calculer la perte adversaire
        loss = criterion_ae(D, X) + self.lambda_e * loss_adv # Calculer la perte de l'autoencodeur
        self.adv_loss += loss_adv.item() # Accumuler la perte adversaire
        self.ae_train_loss += loss.item() # Accumuler la perte de l'autoencodeur
        loss.backward() # Propagation arrière
        optim.step() # Mise à jour des poids
        
        # Mettre à jour lambda_e de manière linéaire jusqu'à LAMBDA_E_max
        if self.iterations<self.total_it:
            self.lambda_e += LAMBDA_E_max / self.total_it
        else:
            self.lambda_e = LAMBDA_E_max
        self.iterations += 1 # Incrémenter le compteur d'itérations
        
class Evaluator:
    """
    Classe responsable de l'évaluation du modèle.
    """
    def __init__(self, ae, latent):
        self.ae = ae # Autoencodeur
        self.latent = latent # discriminateur
        self.latent_evaluate_loss = 0 # Perte totale pour le discriminateur
        self.ae_evaluate_loss = 0 # Perte totale pour l'Autoencodeur
        self.latent_evaluate_accuracy = 0 # Précision totale pour le discriminateur

    def init_parameters(self):
        """Réinitialise les paramètres de validation."""
        self.latent_evaluate_loss = 0
        self.ae_evaluate_loss = 0
        self.latent_evaluate_accuracy = 0


    def latent_step(self, X, y, criterion):
        """
        Évalue le discriminateur latent.
        Args:
            X (torch.Tensor): Données d'entrée.
            y (torch.Tensor): Labels d'attributs.
            criterion (callable): Fonction de perte.
        """
        self.ae.eval()
        self.latent.eval()
        z = self.ae.encoder(X) # Encoder les données
        y_pred = self.latent(z) # Prédire les attributs
        loss = criterion(y_pred, y, False) # Calculer la perte
        self.latent_evaluate_loss += loss.item() # Accumuler la perte du discriminateur
        self.latent_evaluate_accuracy += accuracy_latent(y_pred, y) # Calculer la précision et accumulation de la précision

    def ae_step(self, X, y, criterion):
        """
        Évalue l'autoencodeur.
        Args:
            X (torch.Tensor): Données d'entrée.
            y (torch.Tensor): Labels d'attributs.
            criterion (callable): Fonction de perte.
        Returns:
            torch.Tensor: Reconstruction de l'autoencodeur.
        """
        self.latent.eval()
        self.ae.eval()
        z, D = self.ae(X, y) # Passer les données à travers l'autoencodeur
        loss = criterion(D, X) # Calculer la perte de l'autoencoder
        self.ae_evaluate_loss += loss.item() # Accumuler la perte de l'autoencoder
        return D # Retourner la reconstruction

        


        



