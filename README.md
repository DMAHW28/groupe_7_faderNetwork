# Fader Network Project

This repository contains the implementation of the **Fader Network**, an encoder-decoder architecture designed for image attribute manipulation. The project is structured into several key components to facilitate preprocessing, model training, and evaluation.

## Table of Contents
1. Project Overview
2. Folder Structure
3. Dependencies
4. Installation
5. Usage
6. Modules Description
7. Contributors

---

## Project Overview

The **Fader Network** allows precise manipulation of image attributes (e.g., adding glasses, changing age or gender) by disentangling the latent representation of an image from its attributes. This implementation is based on the original paper _"Manipulating Images by Sliding Attributes"_ and has been designed to evaluate the individual contributions of the encoder, decoder, and discriminator modules.

### Objectives:
- Re-implement the Fader Network architecture.
- Train and evaluate the network using the CelebA dataset.
- Assess the performance of each module (encoder, decoder, and discriminator) by evaluating their respective loss functions.

---

## Folder Structure

```
FaderNetwork/
│
├── .ipynb_checkpoints/      # Notebook checkpoints
├── Data/                    # Scripts and tools for data preprocessing
│   ├── img_align_celeba      # Preprocessing pipeline for the CelebA dataset
    └──list_attr_celeba.txt 
│   └── preprocess.py 
│
├── Models/                  # Fadernetwork modules (encoder, decoder, discriminator)
│   ├── encoder.py
│   ├── decoder.py
│   └── discriminator.py
│
├── Plot/                    # Visualization tools
│   ├── affichage.py         # Plotting and visualization functions
│  
├── Training/                # Training scripts and workflows
│   ├── losses.py             # Contains loss functions for training.
│   ├── train.py              # Main training script
│
└── README.md                # Project documentation
```

---

## Dependencies

This project requires the following Python libraries:
- `torch` (PyTorch)
- `numpy`
- `matplotlib`
- `scikit-learn`
- `PIL`
- Any other dependencies listed in `requirements.txt`

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/FaderNetwork.git
   cd FaderNetwork
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the CelebA dataset and place it in the `Data/` directory.

---

## Usage

### Preprocessing
Run the preprocessing script to prepare the dataset:
```bash
python Data/preprocess.py
```

### Training
Train the model using the provided training script:
```bash
Train.ipynb
```

### Visualization
Use the visualization tools in `Plot/` to generate plots or visualize results:
```bash
Test.ipynb
```

---

## Modules Description

### 1. Data Preprocessing (`Data/`)
- `preprocess.py`: Handles data preparation, including resizing, normalization, and augmentation.

### 2. Models (`Models/`)
- `encoder.py`: Encodes input images into a latent representation.
- `decoder.py`: Decodes the latent representation into an output image.
- `discriminator.py`: Ensures disentanglement between the latent space and attributes using adversarial training.

### 3. Training (`Training/`)
- `train.py`: Main script for orchestrating training.
- `losses.py`: Defines loss functions for categorical attributes and adversarial training.
  
### 4. Plotting and Visualization (`Plot/`)
- `affichage.py`: Provides tools for visualizing training progress and generated images.

---

## Contributors
- **Team Groupe 7**:
  - Khouz360 (Khouzema EMADALY)
  - DavN24 (David NODIER) 
  - DMAHW28 (Marc DRO)
  - nlebel18 (Nathalie LEBEL)

For any questions or feedback, feel free to reach out.
