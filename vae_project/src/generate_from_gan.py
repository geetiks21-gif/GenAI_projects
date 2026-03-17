"""
Standalone Image Generator for GAN — loads a saved Generator and creates new images.
No retraining needed!

Usage:
    python src/generate_from_gan.py --model_dir outputs/gan_saved_model
    python src/generate_from_gan.py --model_dir outputs/gan_saved_model --n_generate 50
    python src/generate_from_gan.py --model_dir outputs/gan_saved_model --n_generate 20 --output outputs/gan_batch2.png
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def load_generator(model_dir):
    """Load the saved generator and config from a model directory."""
    config_path = os.path.join(model_dir, "config.json")
    generator_path = os.path.join(model_dir, "generator.keras")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    if not os.path.exists(generator_path):
        raise FileNotFoundError(f"generator.keras not found in {model_dir}")

    with open(config_path, "r") as f:
        config = json.load(f)

    generator = tf.keras.models.load_model(generator_path)
    channels = config.get('img_channels', 3)
    mode = 'RGB' if channels == 3 else 'Grayscale'
    print(f"\nLoaded GAN generator from: {os.path.abspath(model_dir)}")
    print(f"  Image size : {config['img_height']} x {config['img_width']} x {channels} ({mode})")
    print(f"  Noise dim  : {config['noise_dim']}\n")

    return generator, config


def generate_images(generator, config, n_images, save_path):
    """Generate images using the loaded generator."""
    noise_dim = config["noise_dim"]

    noise = tf.random.normal(shape=(n_images, noise_dim))
    generated = generator(noise, training=False).numpy()

    n_rows = int(np.ceil(n_images / 4))
    fig, axes = plt.subplots(n_rows, 4, figsize=(12, 3 * n_rows))
    axes = axes.flatten()

    for i in range(n_images):
        # Convert from [-1, 1] (tanh output) to [0, 1] for display
        img = (generated[i] + 1.0) / 2.0
        img = np.clip(img, 0, 1)
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


def save_individual_images(generator, config, n_images, output_dir):
    """Save each generated image as a separate file."""
    noise_dim = config["noise_dim"]

    noise = tf.random.normal(shape=(n_images, noise_dim))
    generated = generator(noise, training=False).numpy()

    os.makedirs(output_dir, exist_ok=True)

    for i in range(n_images):
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        img = (generated[i] + 1.0) / 2.0
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.axis('off')
        path = os.path.join(output_dir, f"gan_generated_{i+1:03d}.png")
        fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    print(f"Saved {n_images} individual images to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate images from a saved GAN generator."
    )
    parser.add_argument(
        "--model_dir", type=str, default="outputs/gan_saved_model",
        help="Path to the saved GAN model directory (default: outputs/gan_saved_model)."
    )
    parser.add_argument(
        "--n_generate", type=int, default=16,
        help="Number of images to generate (default: 16)."
    )
    parser.add_argument(
        "--output", type=str, default="outputs/gan_generated.png",
        help="Path to save the generated image grid (default: outputs/gan_generated.png)."
    )
    parser.add_argument(
        "--individual", action="store_true",
        help="Also save each image as a separate file."
    )
    parser.add_argument(
        "--individual_dir", type=str, default="outputs/gan_individual",
        help="Directory for individual images (default: outputs/gan_individual)."
    )
    args = parser.parse_args()

    generator, config = load_generator(args.model_dir)

    generate_images(generator, config, args.n_generate, args.output)

    if args.individual:
        save_individual_images(generator, config, args.n_generate, args.individual_dir)
