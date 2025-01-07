import torch
import matplotlib.pyplot as plt

def plot_attribut(auto_encoder, device, selected_attrs, loader, num_examples=10):
    """
    Affiche num_examples images du loader :
      - Originales
      - Reconstruites
      - Inversées (1 - attributs)
      - Pour chaque attribut, sa version modifiée
    """
    val_images, val_attrs = next(iter(loader))
    val_images = val_images.to(device)
    val_attrs = val_attrs.to(device)

    with torch.no_grad():
        _, reconstructed_images = auto_encoder(val_images, val_attrs)
        _, inversed_images = auto_encoder(val_images, 1 - val_attrs)

    # Paramètres pour l'affichage
    num_cols = 3 + len(selected_attrs)  # Colonnes : Original + Reconstruit + Inversé + x attributs
    num_rows = num_examples            # Lignes : Nombre d'exemples

    plt.figure(figsize=(2 * num_cols, 2 * num_rows))

    titles = ["Originale", "Reconstruites", "Inversée"] + selected_attrs
    for col_idx, title in enumerate(titles):
        plt.subplot(num_rows + 1, num_cols, col_idx + 1)  # +1 pour inclure la ligne de titres
        plt.text(0.5, 0.5, title, ha='center', va='center', fontsize=10)
        plt.axis('off')

    for i in range(num_examples):
        # Image originale
        plt.subplot(num_rows + 1, num_cols, (i + 1) * num_cols + 1)
        plt.imshow(val_images[i].cpu().permute(1, 2, 0) * 0.5 + 0.5)
        plt.axis('off')

        # Image reconstruite
        plt.subplot(num_rows + 1, num_cols, (i + 1) * num_cols + 2)
        plt.imshow(reconstructed_images[i].cpu().permute(1, 2, 0) * 0.5 + 0.5)
        plt.axis('off')

        # Image inversée
        plt.subplot(num_rows + 1, num_cols, (i + 1) * num_cols + 3)
        plt.imshow(inversed_images[i].cpu().permute(1, 2, 0) * 0.5 + 0.5)
        plt.axis('off')

        # Pour chaque attribut, on inverse la paire correspondante
        for k, j in enumerate(range(0, 2 * len(selected_attrs), 2)):
            modified_attrs = val_attrs.clone()
            modified_attrs[i, j:j+2] = 1 - modified_attrs[i, j:j+2]  # Inverser la paire
            # On reconstruit la version modifiée
            with torch.no_grad():
                _, modified_reconstructed_image = auto_encoder(val_images[i:i+1], modified_attrs[i:i+1])
            plt.subplot(num_rows + 1, num_cols, (i + 1) * num_cols + 4 + k)
            plt.imshow(modified_reconstructed_image[0].cpu().permute(1, 2, 0) * 0.5 + 0.5)
            plt.axis('off')

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()



def interact_user(auto_encoder, device, selected_attrs, loader):
    """
    Demande à l'utilisateur de choisir une image (index) dans le batch,
    puis affiche l'image originale, la reconstruite, et propose d'inverser
    un ou plusieurs attributs (ou all).
    """
    val_images, val_attrs = next(iter(loader))
    val_images = val_images.to(device)
    val_attrs = val_attrs.to(device)

    # On demande l'index à l'utilisateur
    max_index = val_images.shape[0] - 1
    img_index = int(input(f"Entrez un index d'image (entre 0 et {max_index}) : "))

    if img_index < 0 or img_index > max_index:
        raise ValueError("L'index saisi est hors limites.")

    # On récupère l'image
    image_to_show = val_images[img_index:img_index+1]
    attrs_to_show = val_attrs[img_index:img_index+1]

    # Reconstruction
    with torch.no_grad():
        _, reconstructed_image = auto_encoder(image_to_show, attrs_to_show)

    original = image_to_show[0].cpu().permute(1, 2, 0) * 0.5 + 0.5
    reconstructed = reconstructed_image[0].cpu().permute(1, 2, 0) * 0.5 + 0.5

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(original.clamp(0, 1))
    axes[0].set_title("Originale")
    axes[0].axis('off')

    axes[1].imshow(reconstructed.clamp(0, 1))
    axes[1].set_title("Reconstruite")
    axes[1].axis('off')
    plt.show()

    # Afficher attributs présents / absents
    attrs_values = attrs_to_show[0].cpu().numpy()  # shape (2*num_attrs,)
    print("\nAttributs pour cette image :")
    for i, attr_name in enumerate(selected_attrs):
        start = 2*i
        pair = attrs_values[start:start+2]
        if (pair[0] == 0 and pair[1] == 1):
            status = "présent"
        else:
            status = "absent"
        print(f" - {attr_name} : {status}")

    print("\nOptions disponibles :")
    print(" - Entrez les noms des attributs à inverser (ex : 'Smiling Eyeglasses')")
    print(" - Entrez 'all' pour inverser tous les attributs")
    print(" - Appuyez sur Entrée sans rien saisir pour quitter")

    attrs_to_invert = input("Quels attributs souhaitez-vous inverser ?\n> ").strip().split()

    # Copie du tenseur
    modified_attrs = attrs_to_show.clone()

    if len(attrs_to_invert) == 1 and attrs_to_invert[0].lower() == "all":
        modified_attrs = 1 - modified_attrs
        print("Tous les attributs ont été inversés.")
    elif len(attrs_to_invert) > 0:
        for attr_name in attrs_to_invert:
            if attr_name in selected_attrs:
                idx_attr = selected_attrs.index(attr_name)
                start = 2 * idx_attr
                end = start + 2
                modified_attrs[0, start:end] = 1 - modified_attrs[0, start:end]
            else:
                print(f"Avertissement: L'attribut '{attr_name}' n'est pas dans la liste {selected_attrs}")
    else:
        print("Aucun attribut inversé. Fin du programme.")
        return

    # Reconstruction avec attributs modifiés
    with torch.no_grad():
        _, modified_image = auto_encoder(image_to_show, modified_attrs)

    modified = modified_image[0].cpu().permute(1, 2, 0) * 0.5 + 0.5

    plt.figure(figsize=(3, 3))
    plt.imshow(modified.clamp(0, 1))
    plt.title("Image modifiée")
    plt.axis('off')
    plt.show()
