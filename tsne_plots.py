# get 5 images from each class
import pandas as pd
import os
import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from PIL import Image
import argparse

def plot_tsnes(model_path):
    # get list of images
    images = []
    labels = []
    all_texts = []
    #class_names = pd.read_csv('diffusion_classifier_clip/prompts/clevr_prompts_cone.csv')['class_name'].tolist()
    path = 'cobi2_datasets/relational'

    for split in os.listdir(path):
        print(split)
        if split != "metadata":
            if split == "train":
                for c in os.listdir(path + "/" + split):
                    if c not in labels and 'left' in c:
                        imgs = os.listdir(path + "/" + split + "/" + c)[:5]
                        imgs = [path + "/" + split + "/" + c + "/" + i for i in imgs]
                        images.extend(imgs)
                        labels.extend([c for i in range(5)])
                    if c not in all_texts:
                        all_texts.append(c)
            else:
                for c in os.listdir(path + "/" + split):
                        c2 = os.listdir(path +"/"+split+ "/" + c)[0]
                        print(c2)
                        if c2 not in labels and 'left' in c2:
                            imgs = os.listdir(path + "/" + split + "/" + c + "/" + c2)[:5]
                            imgs = [path + "/" + split + "/" + c + "/" +c2 + "/" + i for i in imgs]
                            images.extend(imgs)
                            labels.extend([c2 for i in range(5)])
                        if c2 not in all_texts:
                            all_texts.append(c2)
            
    print(all_texts)
    sorted_labels, sorted_images = zip(*sorted(zip(labels, images)))
    images = sorted_images
    labels = sorted_labels



    # Set device (GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the CLIP model
    model_base, transform = clip.load("ViT-B/32", device=device)

    # Clone model for fine-tuned version
    model_finetuned, _ = clip.load("ViT-B/32", device=device)
    model_finetuned.load_state_dict(torch.load(model_path))

    # Initialize lists for embeddings
    image_embeddings_base = []
    image_embeddings_finetuned = []

    # Define marker shapes for different first words in the label
    shape_map = {
        "cube": "s",      # Square
        "sphere": "o",    # Circle
        "cone": "^",      # Triangle
        "cylinder": "P"   # Fat Cross
    }

    # Extract unique third words for color mapping
    third_words = sorted(set(label.split()[2].lower() if len(label.split()) > 2 else "default" for label in labels))
    color_palette = sns.color_palette("Set2", len(third_words))
    color_map = {word: color_palette[i] for i, word in enumerate(third_words)}

    # Process and encode images for both models
    markers = []
    colors = []
    for img_path, label in zip(images, labels):
        image = transform(Image.open(img_path)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features_base = model_base.encode_image(image)
            image_features_finetuned = model_finetuned.encode_image(image)

        image_embeddings_base.append(image_features_base)
        image_embeddings_finetuned.append(image_features_finetuned)
        
        # Determine marker based on the first word in the label
        first_word = label.split()[0].lower()
        markers.append(shape_map.get(first_word, "o"))  # Default to circle if not in shape_map
        
        # Assign colors based on the third word
        third_word = label.split()[2].lower() if len(label.split()) > 2 else "default"
        colors.append(color_map.get(third_word, color_palette[0]))

    # Convert embeddings to NumPy arrays
    image_embeddings_base_np = np.array([torch.flatten(e.cpu()).numpy() for e in image_embeddings_base])
    image_embeddings_finetuned_np = np.array([torch.flatten(e.cpu()).numpy() for e in image_embeddings_finetuned])

    # Perform t-SNE for both sets of embeddings
    tsne = TSNE(n_components=2, perplexity=7, random_state=42)
    tsne_base = tsne.fit_transform(image_embeddings_base_np)
    tsne_finetuned = tsne.fit_transform(image_embeddings_finetuned_np)

    # Configure plot aesthetics
    sns.set(style="white")  # Clean white background, no gridlines

    # Create subplots for side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=100, sharey=True)  # Share y-axis for better comparison

    # Plot base model t-SNE
    for i, marker in enumerate(markers):
        axes[0].scatter(tsne_base[i, 0], tsne_base[i, 1], c=[colors[i]], marker=marker, edgecolors='k', alpha=0.8, s=160)
    axes[0].set_title("Frozen Image Embeddings", fontsize=20, fontweight='bold')
    axes[0].grid(False)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Plot fine-tuned model t-SNE
    for i, marker in enumerate(markers):
        axes[1].scatter(tsne_finetuned[i, 0], tsne_finetuned[i, 1], c=[colors[i]], marker=marker, edgecolors='k', alpha=0.8, s=160)
    axes[1].set_title("Fine-Tuned Image Embeddings", fontsize=20, fontweight='bold')
    axes[1].grid(False)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Create a unified legend combining shape and color
    legend_elements = []
    for shape_name, shape in shape_map.items():
        for word, color in color_map.items():
            if shape_name != word:
                legend_elements.append(Line2D([0], [0], marker=shape, color='w', markerfacecolor=color, markeredgecolor='k', markersize=10, label=f"{shape_name} left {word}"))

    fig.legend(handles=legend_elements, title="Class Labels", loc="center right", bbox_to_anchor=(1.15, 0.55), fontsize=18, title_fontsize=18)

    # Adjust layout to ensure everything fits well
    plt.tight_layout(rect=[0, 0, 0.92, 1])  # Leaves space for the legend on the right

    # Show the plot
    plt.savefig("tsne_images.png")

    # Initialize lists for embeddings
    text_embeddings_base = []
    text_embeddings_finetuned = []

    # Define marker shapes for different first words in the label
    shape_map = {
        "cube": "s",      # Square
        "sphere": "o",    # Circle
        "cone": "^",      # Triangle
        "cylinder": "P"   # Fat Cross
    }

    # Extract unique third words for color mapping
    third_words = sorted(set(label.split()[2].lower() if len(label.split()) > 2 else "default" for label in all_texts))
    color_palette = sns.color_palette("Set2", len(third_words))
    color_map = {word: color_palette[i] for i, word in enumerate(third_words)}

    # Process and encode images for both models
    markers = []
    colors = []
    done_labels = []
    for label in all_texts:
        print(label)
        image = transform(Image.open(img_path)).unsqueeze(0).to(device)
        text = clip.tokenize(label).to(device)
        with torch.no_grad():
            text_base = model_base.encode_text(text)
            text_finetuned = model_finetuned.encode_text(text)

        if label not in done_labels:
            text_embeddings_base.append(text_base)
            text_embeddings_finetuned.append(text_finetuned)
            done_labels.append(label)

            # Determine marker based on the first word in the label
            first_word = label.split()[0].lower()
            markers.append(shape_map.get(first_word, "o"))  # Default to circle if not in shape_map

            # Assign colors based on the third word
            third_word = label.split()[2].lower() if len(label.split()) > 2 else "default"
            colors.append(color_map.get(third_word, color_palette[0]))

    # Convert embeddings to NumPy arrays
    text_embeddings_base_np = np.array([torch.flatten(e.cpu()).numpy() for e in text_embeddings_base])
    text_embeddings_finetuned_np = np.array([torch.flatten(e.cpu()).numpy() for e in text_embeddings_finetuned])

    # Perform t-SNE for both sets of embeddings
    tsne = TSNE(n_components=2, perplexity=7, random_state=42)
    tsne_base = tsne.fit_transform(text_embeddings_base_np)
    tsne_finetuned = tsne.fit_transform(text_embeddings_finetuned_np)

    # Configure plot aesthetics
    sns.set(style="white")  # Clean white background, no gridlines

    # Create subplots for side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=100, sharey=True)  # Share y-axis for better comparison

    # Plot base model t-SNE
    for i, marker in enumerate(markers):
        axes[0].scatter(tsne_base[i, 0], tsne_base[i, 1],
                        c=[colors[i]], marker=marker,
                        edgecolors='k', alpha=0.8, s=160)
    axes[0].set_title("Frozen Text Embeddings", fontsize=20, fontweight='bold')
    axes[0].grid(False)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Plot fine-tuned model t-SNE
    for i, marker in enumerate(markers):
        axes[1].scatter(tsne_finetuned[i, 0], tsne_finetuned[i, 1],
                        c=[colors[i]], marker=marker,
                        edgecolors='k', alpha=0.8, s=160)
    axes[1].set_title("Fine-Tuned Text Embeddings", fontsize=20, fontweight='bold')
    axes[1].grid(False)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    legend_elements = []
    for label in all_texts:
        parts = label.split()
        if len(parts) >= 3:
            first_word, second_word, third_word = parts[0].lower(), parts[1].lower(), parts[2].lower()
            shape = shape_map.get(first_word, "o")
            color = color_map.get(third_word, color_palette[0])
            legend_label = f"{first_word} {second_word} {third_word}"

            # Avoid duplicates in legend
            if legend_label not in [h.get_label() for h in legend_elements]:
                legend_elements.append(
                    Line2D([0], [0], marker=shape, color='w',
                        markerfacecolor=color, markeredgecolor='k',
                        markersize=10, label=legend_label)
                )

    # Place legend at the bottom
    fig.legend(handles=legend_elements,
            title="Class Labels",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.4),
            fontsize=14,
            title_fontsize=16,
            ncol=3)

    # Adjust layout to make space for bottom legend
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # Save plot
    plt.savefig("tsne_text.png", bbox_inches="tight")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path to finetuned CLIP')
    args = parser.parse_args()
    plot_tsnes(args.model_path)