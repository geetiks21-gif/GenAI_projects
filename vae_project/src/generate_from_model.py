"""
Standalone Image Generator — loads a saved VAE decoder and generates new images.
No retraining required!

Usage:
    python generate_from_model.py --model_dir outputs/saved_model
    python generate_from_model.py --model_dir outputs/saved_model --n_generate 50
    python generate_from_model.py --model_dir outputs/saved_model --n_generate 100 --output outputs/batch2.png
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def load_decoder(model_dir):
    """Load the saved decoder and config from a model directory."""
    config_path = os.path.join(model_dir, "config.json")
    decoder_path = os.path.join(model_dir, "decoder.keras")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    if not os.path.exists(decoder_path):
        raise FileNotFoundError(f"decoder.keras not found in {model_dir}")

    with open(config_path, "r") as f:
        config = json.load(f)

    decoder = tf.keras.models.load_model(decoder_path)
    print(f"\nLoaded decoder from: {os.path.abspath(model_dir)}")
    channels = config.get('img_channels', 1)
    mode = 'RGB' if channels == 3 else 'Grayscale'
    print(f"  Image size : {config['img_height']} x {config['img_width']} x {channels} ({mode})")
    print(f"  Latent dim : {config['latent_dim']}\n")

    return decoder, config


def generate_images(decoder, config, n_images, save_path):
    """Generate images using the loaded decoder."""
    latent_dim = config["latent_dim"]
    img_height = config["img_height"]
    img_width = config["img_width"]

    random_latent_vectors = tf.random.normal(shape=(n_images, latent_dim))
    generated_images = decoder(random_latent_vectors).numpy()

    n_rows = int(np.ceil(n_images / 4))

    fig, axes = plt.subplots(n_rows, 4, figsize=(12, 3 * n_rows))
    axes = axes.flatten()
    channels = config.get('img_channels', 1)
    for i in range(n_images):
        img = np.clip(generated_images[i], 0, 1)
        if channels == 1:
            axes[i].imshow(img.squeeze(), cmap='gray')
        else:
            axes[i].imshow(img)
        axes[i].axis('off')
    for j in range(n_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Generated {n_images} images saved to: {os.path.abspath(save_path)}")

    plt.show(block=False)
    plt.close(fig)


def save_individual_images(decoder, config, n_images, output_dir):
    """Save each generated image as a separate file."""
    latent_dim = config["latent_dim"]

    random_latent_vectors = tf.random.normal(shape=(n_images, latent_dim))
    generated_images = decoder(random_latent_vectors).numpy()

    os.makedirs(output_dir, exist_ok=True)

    channels = config.get('img_channels', 1)
    for i in range(n_images):
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        img = np.clip(generated_images[i], 0, 1)
        if channels == 1:
            ax.imshow(img.squeeze(), cmap='gray')
        else:
            ax.imshow(img)
        ax.axis('off')
        path = os.path.join(output_dir, f"generated_{i+1:03d}.png")
        fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    print(f"Saved {n_images} individual images to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate images from a saved VAE model (no retraining needed)."
    )
    parser.add_argument(
        "--model_dir", type=str, default="outputs/saved_model",
        help="Path to directory with saved encoder.keras, decoder.keras, config.json."
    )
    parser.add_argument(
        "--n_generate", type=int, default=16,
        help="Number of images to generate (default: 16)."
    )
    parser.add_argument(
        "--output", type=str, default="outputs/generated_images.png",
        help="Path to save the generated image grid (default: outputs/generated_images.png)."
    )
    parser.add_argument(
        "--save_individual", action="store_true",
        help="Also save each image as a separate file in outputs/individual/."
    )
    args = parser.parse_args()

    decoder, config = load_decoder(args.model_dir)

    generate_images(decoder, config, args.n_generate, args.output)

    if args.save_individual:
        save_individual_images(decoder, config, args.n_generate, "outputs/individual")
