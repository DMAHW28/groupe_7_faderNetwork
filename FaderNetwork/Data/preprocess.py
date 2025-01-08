import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, Dataset

N_IMAGES = 202599  # Nombre total d'images dans le dataset CelebA
IMG_SIZE = 256  # # Taille cible pour redimensionner les images (256x256 pixels)
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data')  # Chemin d'accès aux données
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # matériel disponible (GPU ou CPU)


def preprocess_images_txt(data_index):
    """
    Prépare la liste des chemins d'accès aux images pour un intervalle donné.
    Args:
        data_index (tuple): intervalle d'images.
    Returns:
        list: Liste de nom des fichiers images dans l'intervalle donné.
    """
    start, stop = data_index
    all_images = []
    for i in tqdm(range(start + 1, stop + 1), desc="Processing in batches"):  # Parcourt les indices d'image
        image_path = f'img_align_celeba/{i:06}.jpg'
        all_images.append(image_path)
    return all_images


def preprocess_attributes(data_index):
    """
    Prépare les attributs des images dans l'intervalle donné.
    Les attributs sont codés en [0, 1] pour la valeur "1" et [1, 0] pour la valeur "0".
    Args:
        data_index (tuple): intervalle d'images.
    Returns:
        torch.Tensor: tenseur des attributs codés pour les images dans l'intervalle donné.
    """
    start, stop = data_index
    n_images = stop - start

    # Chargement des attributs depuis un fichier texte
    attr_lines = [line.rstrip() for line in open(os.path.join(DATA_PATH, 'list_attr_celeba.txt'), 'r')]
    attr_keys = attr_lines[1].split()  # Liste des noms des attributs
    attr_position = {}  # Dictionnaire pour stocker la position de chaque attribut
    attributes = np.zeros((n_images, 2 * len(attr_keys)), dtype=int)  # Matrice pour encoder les attributs
    for i, line in tqdm(enumerate(attr_lines[2 + start:stop]), desc="Processing Attribute",
                        total=len(attr_lines[2 + start:stop])):
        split = line.split()
        for j, value in enumerate(split[1:]):  # Itération sur les valeurs des attributs
            if value == '1':
                attributes[i, 2 * j] = 0
                attributes[i, 2 * j + 1] = 1
            else:
                attributes[i, 2 * j] = 1
                attributes[i, 2 * j + 1] = 0

            if i == 0:  # Stockage des positions des attributs
                attr_position[attr_keys[j]] = 2 * j
    # Conversion en tenseur PyTorch        
    attributes = torch.tensor(attributes, dtype=torch.float32, device=device)
    return attributes, attr_position


class CelebADataset(Dataset):
    """
    Classe pour gérer le dataset CelebA en chargeant les images et les attributs associés.
    """

    def __init__(self, images, attributes, attributes_position):
        self.images = images
        self.attributes = attributes
        self.attributes_position = attributes_position

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Chargement et prétraitement de l'image
        img_path = self.images[idx]
        image = cv2.imread(os.path.join(DATA_PATH, img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Conversion BGR -> RGB
        image = image[20:-20]  # Découpe pour éliminer les bords
        resized_image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)  # Redimensionnement
        tensor_image = torch.tensor(resized_image, dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0
        tensor_image = 2 * tensor_image - 1  # Normalisation des pixels entre [-1, 1]
        # Chargement des attributs
        attrs = torch.stack([self.attributes[idx, start:end + 1] for start, end in self.attributes_position],
                            dim=0).flatten()
        return tensor_image, attrs


def load_images(index, selected_attrs, batch_size=32):
    """
    Charge les images et les attributs pour un intervalle donné.
    Args:
        index (tuple): intervalle d'images.
        selected_attrs(list(string)): liste des noms des attributs utilisé pour l'apprentissage.
    Returns:
        dataloader: Liste des DataLoaders train/validation/test images et attributs
    """
    images = preprocess_images_txt(index)
    attributes, attr_position = preprocess_attributes(index)
    position = [(attr_position[satt], attr_position[satt] + 1) for satt in selected_attrs]
    images, attributes = split_data_for_learning(images, attributes)
    base_data_loader = create_data_loader(images, attributes, position, batch_size=batch_size)
    return base_data_loader


def split_data_for_learning(images, attributes):
    """
    Divise les données en trois ensembles : 50% pour l'entraînement, 25% pour la validation, 25% pour les tests.
    Args:
        images (list): Liste de nom des fichiers images dans l'intervalle donné..
        attributes (torch.Tensor): tenseur des attributs codés pour les images.
    Returns:
        images(tuple): tuple de liste des nom de fichier images pour les bases train/validation/test 
        attributes(tuple): tuple de liste des tenseurs des attributs codés pour les bases train/validation/test 
    """
    example_size = len(images)
    train_index = example_size // 2
    valid_index = (3 * example_size) // 4
    test_index = example_size
    # Division des images et des attributs
    train_images = images[:train_index]
    valid_images = images[train_index:valid_index]
    test_images = images[valid_index:test_index]
    train_attributes = attributes[:train_index]
    valid_attributes = attributes[train_index:valid_index]
    test_attributes = attributes[valid_index:test_index]
    images = train_images, valid_images, test_images
    attributes = train_attributes, valid_attributes, test_attributes
    return images, attributes


def create_data_loader(images, attributes, position, batch_size=32):
    """
    Crée des DataLoaders pour les ensembles d'entraînement, de validation et de test.
    Args:
        images (tuple): Images divisées (train, validation, test).
        attributes (tuple): Attributs divisés (train, validation, test).
        position (list): Positions des attributs sélectionnés.
        batch_size (int): Taille des batchs.
    
    Returns:
        list: Liste des DataLoaders (train, validation, test).
    """
    base_data_loader = []
    for i, (X, y) in enumerate(zip(images, attributes)):
        data_set = CelebADataset(X, y, position)
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=(i == 0))  # Shuffle uniquement pour l'entraînement
        base_data_loader.append(data_loader)
    return base_data_loader


def create_data_file(selected_attrs, step=10000, batch_size=32, n_img=N_IMAGES):
    """
    Charge les données par étapes pour accélérer le traitement.
    Args:
        selected_attrs (list): Liste des noms des attributs sélectionnés.
        step (int): Taille de chaque intervalle d'images.
        batch_size (int): Taille des batchs.
        n_img (int): Nombre total d'images.
    
    Yields:
        list: Liste des DataLoaders pour chaque intervalles.
    """
    indexes = [(i, min(i + step, n_img)) for i in range(0, n_img, step)]  # Création des intervalles

    for i, index in enumerate(indexes):
        base_data_loader = load_images(index=index, selected_attrs=selected_attrs, batch_size=batch_size)
        yield base_data_loader # Génère les DataLoaders pour chaque intervalles



