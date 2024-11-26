import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

N_IMAGES = 202599
IMG_SIZE = 256
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data')

def preprocess_images(data_num, data_index, batch_size=1000):
    IMG_PATH = f'images_{data_num}.pth'
    start, stop = data_index
    if os.path.isfile(os.path.join(DATA_PATH, IMG_PATH)):
        print(f"{IMG_PATH} exists, nothing to do.")
        return

    print(f"Processing images from index {start + 1} to {stop} ...")
    all_images = []
    for batch_start in tqdm(range(start + 1, stop + 1, batch_size), desc="Processing in batches"):
        batch_end = min(batch_start + batch_size, stop + 1)
        raw_images = []
        for i in range(batch_start, batch_end):
            image_path = f'img_align_celeba/{i:06}.jpg'
            image = cv2.imread(os.path.join(DATA_PATH, image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            raw_images.append(image[20:-20])
        for image in raw_images:
            resized_image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            tensor_image = torch.tensor(resized_image, dtype=torch.float32).permute(2, 0, 1) / 255.0
            all_images.append(tensor_image)
    data = torch.stack(all_images)

    print(f"Saving images to {IMG_PATH} ...")
    torch.save(data, os.path.join(DATA_PATH, IMG_PATH))
    # return data

def preprocess_attributes(data_num, data_index):
    ATTR_PATH = f'attributes_{data_num}.pth'
    if os.path.isfile(os.path.join(DATA_PATH, ATTR_PATH)):
        print("%s exists, nothing to do." % ATTR_PATH)
        return
    start, stop = data_index
    n_images = stop - start
    attr_lines = [line.rstrip() for line in open(os.path.join(DATA_PATH, 'list_attr_celeba.txt'), 'r')]
    attr_keys = attr_lines[1].split()
    attributes = np.zeros((n_images, len(attr_keys)), dtype=bool)
    print(f"Processing attributes from index {start + 1} to {stop} ...")
    for i, line in tqdm(enumerate(attr_lines[2+start:stop]), desc="Processing Attibute"):
        split = line.split()
        for j, value in enumerate(split[1:]):
            attributes[i, j] = value == '1'
    attributes = torch.tensor(attributes.astype(int), dtype=torch.float32)
    print("Saving attributes to %s ..." % ATTR_PATH)
    torch.save(attributes, os.path.join(DATA_PATH, ATTR_PATH))
    # return torch.tensor(attributes.astype(int), dtype=torch.float32)

def load_images(data_num, batch_size):
    """
    Load celebA dataset.
    """
    # load data
    images = torch.load(os.path.join(DATA_PATH, f'images_{data_num}.pth'), weights_only=True)
    attributes = torch.load(os.path.join(DATA_PATH, f'attributes_{data_num}.pth'), weights_only=True)
    images, attributes = split_data_for_learning(images, attributes)
    # display_images(images[0])
    base_data_loader = create_data_loader(images, attributes, batch_size=batch_size)
    return base_data_loader

def split_data_for_learning(images, attributes):
    train_index = 5000
    valid_index = 7500
    test_index = 10000
    train_images = images[:train_index]
    valid_images = images[train_index:valid_index]
    test_images = images[valid_index:test_index]
    train_attributes = attributes[:train_index]
    valid_attributes = attributes[train_index:valid_index]
    test_attributes = attributes[valid_index:test_index]
    images = train_images, valid_images, test_images
    attributes = train_attributes, valid_attributes, test_attributes
    return images, attributes

def create_data_loader(images, attributes, batch_size=32):
    base_data_loader = []
    for X, y in zip(images, attributes):
        data_set = TensorDataset(X, y)
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
        base_data_loader.append(data_loader)
    return base_data_loader


def create_data_file():
    step = 20000
    batch_size = 1000
    indexes = [(i, min(i + step, N_IMAGES)) for i in range(0, N_IMAGES, step)]

    for i, index in enumerate(indexes):
        preprocess_images(i, index, batch_size=batch_size)
        preprocess_attributes(i, index)

"""
step = 10000
index = (0, step)
preprocess_images(0, index)
preprocess_attributes(0, index)
"""