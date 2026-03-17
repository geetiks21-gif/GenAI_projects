"""
Deep Convolutional GAN (DCGAN) with TensorFlow
for Image Generation Using Custom Image Folders

A GAN has TWO neural networks competing against each other:
    - Generator:     Creates fake images from random noise (tries to fool the discriminator)
    - Discriminator: Tries to tell apart real images from fake ones

They play a "cat and mouse" game:
    - The Generator gets better at creating realistic images
    - The Discriminator gets better at spotting fakes
    - Over time, the Generator produces images that look real!

Usage:
    python src/gan_image_generation.py --data_dir sample_images
    python src/gan_image_generation.py --data_dir sample_images --epochs 200 --batch_size 32

Steps:
    1. Import the Necessary Libraries
    2. Load Images from a Custom Folder
    3. Set Hyperparameters
    4. Build the Generator (noise -> fake image)
    5. Build the Discriminator (image -> real or fake?)
    6. Define Loss Functions and Optimizers
    7. Define the Training Step
    8. Train the GAN
    9. Save the Trained Generator
    10. Generate New Images
"""

# =============================================================================
# Step 1: Import the Necessary Libraries
# =============================================================================
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

# =============================================================================
# Image Configuration Constants
# =============================================================================
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3  # RGB color
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp')


# =============================================================================
# Step 2: Load Images from a Custom Folder
# =============================================================================
# This is the same loader from the VAE project. We load all images, resize them,
# normalize pixel values to [-1, 1] (tanh range, standard for GANs), and
# optionally augment (flip + rotations) to increase training data.
# =============================================================================
def load_images_from_folder(folder_path, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, augment=True):
    """
    Load all images from a folder (recursively), convert to RGB, resize,
    normalize to [-1, 1] (for tanh output), and apply data augmentation.

    Why [-1, 1] instead of [0, 1]?
        GANs use tanh activation in the generator's last layer, which outputs
        values in [-1, 1]. So we match the real data to the same range.

    Args:
        folder_path: Path to folder containing image files.
        img_height:  Target image height (default: 128).
        img_width:   Target image width  (default: 128).
        augment:     If True, apply augmentation (flip + rotations) for ~5x data.

    Returns:
        dataset: NumPy array of shape (N, img_height, img_width, 3), values in [-1, 1].
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Data directory not found: {folder_path}")

    images = []
    skipped = 0
    original_count = 0

    for root, _, filenames in os.walk(folder_path):
        for filename in sorted(filenames):
            if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
                continue
            filepath = os.path.join(root, filename)
            try:
                img = Image.open(filepath).convert('RGB')
                img = img.resize((img_width, img_height))
                img_array = np.array(img, dtype=np.uint8)  # uint8 saves 4x memory vs float32
                images.append(img_array)
                original_count += 1

                # Data augmentation: flip + rotations = 5x total
                if augment:
                    images.append(np.fliplr(img_array))      # horizontal flip
                    images.append(np.rot90(img_array, k=1))   # 90°
                    images.append(np.rot90(img_array, k=2))   # 180°
                    images.append(np.rot90(img_array, k=3))   # 270°

            except Exception as e:
                print(f"  [SKIP] {filename}: {e}")
                skipped += 1

    if len(images) == 0:
        raise ValueError(
            f"No valid images found in '{folder_path}'.\n"
            f"Supported formats: {SUPPORTED_EXTENSIONS}"
        )

    # Normalize to [-1, 1] for tanh (GAN standard)
    # Formula: (pixel / 127.5) - 1  maps [0, 255] -> [-1, 1]
    # Convert uint8 → float32 and normalize in one step to avoid double memory
    data = np.array(images, dtype=np.float32)
    del images  # free the list immediately
    data = data / 127.5 - 1.0
    np.random.shuffle(data)

    print(f"\nLoaded {original_count} original images from: {folder_path}")
    if augment:
        print(f"  Augmented (flip + rotations): {original_count} -> {len(data)} images")
    if skipped:
        print(f"  Skipped {skipped} unreadable files")
    print(f"  Image size : {img_height} x {img_width} x {IMG_CHANNELS} (RGB)")
    print(f"  Pixel range: [-1, 1] (tanh)\n")

    return data


# =============================================================================
# Step 3: Set Hyperparameters
# =============================================================================
# noise_dim: The size of the random noise vector fed to the Generator.
#   Think of it as the "seed" — each random vector produces a different image.
#   Typical values: 100-256.
#
# learning_rate: How fast the networks learn. GANs are sensitive to this;
#   too high and training collapses, too low and it takes forever.
# =============================================================================
noise_dim = 128       # dimensionality of the random noise input
learning_rate = 0.0002
beta_1 = 0.5          # Adam beta1 — lower than default (0.9) for GAN stability


# =============================================================================
# Step 4: Build the Generator
# =============================================================================
# The Generator is like a "forger" — it takes random noise and transforms it
# into an image, trying to make it look as realistic as possible.
#
# Architecture (noise → image):
#   Random noise (128,)
#     → Dense + Reshape to 8x8x512  (small feature map)
#     → Conv2DTranspose 8x8→16x16   (upsample)
#     → Conv2DTranspose 16x16→32x32 (upsample)
#     → Conv2DTranspose 32x32→64x64 (upsample)
#     → Conv2DTranspose 64x64→128x128 (upsample)
#     → Conv2D 128x128x3 (final RGB image with tanh activation)
#
# Key design choices:
#   - BatchNormalization: stabilizes training, applied after every layer
#   - LeakyReLU: prevents "dying neurons" (better than regular ReLU for GANs)
#   - tanh activation at the end: outputs pixel values in [-1, 1]
# =============================================================================
def build_generator(noise_dim):
    """Build the Generator network: random noise → fake RGB image."""

    model = tf.keras.Sequential(name='generator')

    # Layer 1: Dense — project noise to a small spatial feature map
    # Takes the noise vector and expands it to 8*8*512 = 32768 values
    model.add(tf.keras.layers.Dense(8 * 8 * 512, use_bias=False, input_shape=(noise_dim,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(negative_slope=0.2))
    model.add(tf.keras.layers.Reshape((8, 8, 512)))
    # Shape: (8, 8, 512)

    # Layer 2: Upsample 8x8 → 16x16
    # Conv2DTranspose is "reverse convolution" — it INCREASES spatial size
    model.add(tf.keras.layers.Conv2DTranspose(256, 4, strides=2, padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(negative_slope=0.2))
    # Shape: (16, 16, 256)

    # Layer 3: Upsample 16x16 → 32x32
    model.add(tf.keras.layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(negative_slope=0.2))
    # Shape: (32, 32, 128)

    # Layer 4: Upsample 32x32 → 64x64
    model.add(tf.keras.layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(negative_slope=0.2))
    # Shape: (64, 64, 64)

    # Layer 5: Upsample 64x64 → 128x128
    model.add(tf.keras.layers.Conv2DTranspose(32, 4, strides=2, padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(negative_slope=0.2))
    # Shape: (128, 128, 32)

    # Output Layer: 128x128x32 → 128x128x3 (RGB image)
    # tanh activation → pixel values in [-1, 1]
    model.add(tf.keras.layers.Conv2D(IMG_CHANNELS, 3, padding='same', activation='tanh'))
    # Shape: (128, 128, 3)

    return model


# =============================================================================
# Step 5: Build the Discriminator
# =============================================================================
# The Discriminator is like a "detective" — it looks at an image and decides
# whether it's REAL (from the dataset) or FAKE (created by the Generator).
#
# Architecture (image → real/fake score):
#   RGB image (128, 128, 3)
#     → Conv2D 128x128→64x64     (downsample)
#     → Conv2D 64x64→32x32       (downsample)
#     → Conv2D 32x32→16x16       (downsample)
#     → Conv2D 16x16→8x8         (downsample)
#     → Flatten + Dense → 1 value (real or fake?)
#
# Key design choices:
#   - NO BatchNormalization (or only partial) — DCGAN paper recommends
#     skipping BN in discriminator's first layer
#   - LeakyReLU with 0.2 slope — standard for discriminators
#   - Dropout: prevents the discriminator from becoming too strong too fast
#   - Output is a single number (logit) — NOT sigmoid! We use
#     "from_logits=True" in the loss function for numerical stability.
# =============================================================================
def build_discriminator(img_height, img_width):
    """Build the Discriminator network: RGB image → real/fake score."""

    model = tf.keras.Sequential(name='discriminator')

    # Layer 1: 128x128 → 64x64 (NO BatchNorm in first layer — DCGAN convention)
    model.add(tf.keras.layers.Conv2D(64, 4, strides=2, padding='same',
                                     input_shape=(img_height, img_width, IMG_CHANNELS)))
    model.add(tf.keras.layers.LeakyReLU(negative_slope=0.2))
    model.add(tf.keras.layers.Dropout(0.3))
    # Shape: (64, 64, 64)

    # Layer 2: 64x64 → 32x32
    model.add(tf.keras.layers.Conv2D(128, 4, strides=2, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(negative_slope=0.2))
    model.add(tf.keras.layers.Dropout(0.3))
    # Shape: (32, 32, 128)

    # Layer 3: 32x32 → 16x16
    model.add(tf.keras.layers.Conv2D(256, 4, strides=2, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(negative_slope=0.2))
    model.add(tf.keras.layers.Dropout(0.3))
    # Shape: (16, 16, 256)

    # Layer 4: 16x16 → 8x8
    model.add(tf.keras.layers.Conv2D(512, 4, strides=2, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(negative_slope=0.2))
    # Shape: (8, 8, 512)

    # Output: Flatten → single score
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))  # Raw logit (no sigmoid!)
    # Output: (1,) — a single real/fake score

    return model


# =============================================================================
# Step 6: Define Loss Functions and Optimizers
# =============================================================================
# GAN loss is a minimax game:
#   - Discriminator wants to MAXIMIZE its ability to classify real vs fake
#   - Generator wants to MINIMIZE the discriminator's ability (i.e., fool it)
#
# We use Binary Cross Entropy (BCE) with logits:
#   - Real images should get label 1 (discriminator says "real")
#   - Fake images should get label 0 (discriminator says "fake")
#   - Generator wants fake images to get label 1 (fool the discriminator)
#
# Label smoothing: Instead of hard labels (0 and 1), we use soft labels
#   (0.0 and 0.9) to prevent the discriminator from becoming overconfident.
# =============================================================================
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    """
    Discriminator loss: wants to correctly classify real images as 1 and fake as 0.

    - real_loss: How well it identifies real images (target = 0.9 for label smoothing)
    - fake_loss: How well it identifies fake images (target = 0.0)
    """
    real_loss = cross_entropy(tf.ones_like(real_output) * 0.9, real_output)  # label smoothing
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


def generator_loss(fake_output):
    """
    Generator loss: wants the discriminator to think fake images are REAL.

    - The generator wants fake_output to be classified as 1 (real)
    - If the discriminator says "fake" (output near 0), loss is HIGH
    - If the discriminator says "real" (output near 1), loss is LOW
    """
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# =============================================================================
# Step 7: Define the Training Step
# =============================================================================
# Each training step:
#   1. Generator creates a batch of fake images from random noise
#   2. Discriminator sees both real images AND fake images
#   3. Discriminator loss: "Did I correctly spot real vs fake?"
#   4. Generator loss: "Did my fakes fool the discriminator?"
#   5. Both networks update their weights (backpropagation)
#
# IMPORTANT: We train both networks SIMULTANEOUSLY. This is what creates
# the adversarial "competition" that drives improvement.
# =============================================================================
@tf.function  # Compiles to a TensorFlow graph for speed
def train_step(real_images, generator, discriminator, gen_optimizer, disc_optimizer, batch_size, noise_dim):
    """
    One training step for the GAN.

    Args:
        real_images:    A batch of real images from the dataset.
        generator:      The Generator model.
        discriminator:  The Discriminator model.
        gen_optimizer:  Optimizer for the Generator.
        disc_optimizer: Optimizer for the Discriminator.
        batch_size:     Number of images in this batch.
        noise_dim:      Dimensionality of the noise vector.

    Returns:
        gen_loss, disc_loss: The losses for this step.
    """
    # Step 7a: Create random noise — this is the Generator's input
    noise = tf.random.normal([batch_size, noise_dim])

    # We need to record gradients for BOTH networks separately
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Step 7b: Generator creates fake images
        generated_images = generator(noise, training=True)

        # Step 7c: Discriminator evaluates both real and fake images
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        # Step 7d: Calculate losses
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # Step 7e: Calculate gradients (which direction to adjust weights)
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Step 7f: Update weights using the gradients
    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    return gen_loss, disc_loss


# =============================================================================
# Step 8: Train the GAN
# =============================================================================
def train_gan(dataset, generator, discriminator, gen_optimizer, disc_optimizer,
              epochs, batch_size, noise_dim, preview_interval=10):
    """
    Full GAN training loop.

    What happens each epoch:
        1. Shuffle the dataset
        2. Split into batches
        3. Run train_step on each batch
        4. Print losses
        5. Optionally save a preview of generated images

    Args:
        dataset:          NumPy array of training images.
        generator:        The Generator model.
        discriminator:    The Discriminator model.
        gen_optimizer:    Optimizer for the Generator.
        disc_optimizer:   Optimizer for the Discriminator.
        epochs:           Number of training epochs.
        batch_size:       Training batch size.
        noise_dim:        Dimensionality of the noise vector.
        preview_interval: Save preview images every N epochs.
    """
    # Convert numpy array to TensorFlow dataset for efficient batching
    tf_dataset = tf.data.Dataset.from_tensor_slices(dataset)
    tf_dataset = tf_dataset.shuffle(buffer_size=len(dataset)).batch(batch_size, drop_remainder=True)

    # Fixed noise vector for tracking progress — same noise each time
    # so we can see how the generator improves over epochs
    seed_noise = tf.random.normal([16, noise_dim])

    os.makedirs("outputs/gan_progress", exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  GAN Training — {epochs} epochs, batch_size={batch_size}")
    print(f"  Dataset: {len(dataset)} images | Noise dim: {noise_dim}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        epoch_gen_loss = 0.0
        epoch_disc_loss = 0.0
        num_batches = 0

        for batch in tf_dataset:
            actual_batch_size = tf.shape(batch)[0]
            g_loss, d_loss = train_step(
                batch, generator, discriminator,
                gen_optimizer, disc_optimizer,
                actual_batch_size, noise_dim
            )
            epoch_gen_loss += g_loss
            epoch_disc_loss += d_loss
            num_batches += 1

        # Average losses over all batches
        avg_gen = epoch_gen_loss / num_batches
        avg_disc = epoch_disc_loss / num_batches

        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{epochs}  |  "
                  f"Generator Loss: {avg_gen:.4f}  |  "
                  f"Discriminator Loss: {avg_disc:.4f}")

        # Save preview images periodically
        if epoch % preview_interval == 0 or epoch == 1:
            save_preview(generator, seed_noise, epoch)

    print(f"\nTraining complete!")


def save_preview(generator, seed_noise, epoch):
    """Save a grid of generated images to track training progress."""
    generated = generator(seed_noise, training=False).numpy()

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        # Convert from [-1, 1] back to [0, 1] for display
        img = (generated[i] + 1.0) / 2.0
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.axis('off')

    plt.suptitle(f'Epoch {epoch}', fontsize=14)
    plt.tight_layout()
    path = f"outputs/gan_progress/epoch_{epoch:04d}.png"
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Step 9 & 10: Save Model and Generate Images
# =============================================================================
def save_gan_model(generator, discriminator, model_dir, noise_dim):
    """Save the trained generator and discriminator along with config."""
    os.makedirs(model_dir, exist_ok=True)
    generator.save(os.path.join(model_dir, "generator.keras"))
    discriminator.save(os.path.join(model_dir, "discriminator.keras"))

    config = {
        "img_height": IMG_HEIGHT,
        "img_width": IMG_WIDTH,
        "img_channels": IMG_CHANNELS,
        "noise_dim": noise_dim,
        "model_type": "DCGAN"
    }
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nGAN model saved to: {os.path.abspath(model_dir)}")
    print(f"  - generator.keras")
    print(f"  - discriminator.keras")
    print(f"  - config.json")


def generate_images(generator, noise_dim, n_images, save_path="outputs/gan_generated.png"):
    """Generate images by feeding random noise into the trained Generator."""
    noise = tf.random.normal([n_images, noise_dim])
    generated = generator(noise, training=False).numpy()

    n_rows = int(np.ceil(n_images / 4))
    fig, axes = plt.subplots(n_rows, 4, figsize=(12, 3 * n_rows))
    axes = axes.flatten()

    for i in range(n_images):
        # Convert from [-1, 1] back to [0, 1] for display
        img = (generated[i] + 1.0) / 2.0
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].axis('off')

    for j in range(n_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Generated {n_images} images saved to: {os.path.abspath(save_path)}")
    plt.show(block=False)
    plt.close(fig)


# =============================================================================
# Main: Parse arguments, load data, build GAN, train, and generate
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a DCGAN on your own images and generate new ones."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to the folder containing training images."
    )
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="Number of training epochs (default: 200)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Training batch size (default: 32)."
    )
    parser.add_argument(
        "--noise_dim", type=int, default=noise_dim,
        help=f"Noise vector dimensionality (default: {noise_dim})."
    )
    parser.add_argument(
        "--n_generate", type=int, default=16,
        help="Number of images to generate after training (default: 16)."
    )
    parser.add_argument(
        "--model_dir", type=str, default="outputs/gan_saved_model",
        help="Directory to save trained GAN (default: outputs/gan_saved_model)."
    )
    parser.add_argument(
        "--no_augment", action="store_true",
        help="Disable data augmentation (default: augmentation is ON)."
    )
    parser.add_argument(
        "--preview_interval", type=int, default=10,
        help="Save progress images every N epochs (default: 10)."
    )
    args = parser.parse_args()

    noise_dim = args.noise_dim

    # --- Step 2: Load images ---
    dataset = load_images_from_folder(args.data_dir, augment=not args.no_augment)

    # --- Step 4: Build Generator ---
    generator = build_generator(noise_dim)
    generator.summary()

    # --- Step 5: Build Discriminator ---
    discriminator = build_discriminator(IMG_HEIGHT, IMG_WIDTH)
    discriminator.summary()

    # --- Step 6: Create Optimizers ---
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1)

    # --- Step 8: Train the GAN ---
    train_gan(
        dataset, generator, discriminator,
        gen_optimizer, disc_optimizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        noise_dim=noise_dim,
        preview_interval=args.preview_interval
    )

    # --- Step 9: Save the trained model ---
    save_gan_model(generator, discriminator, args.model_dir, noise_dim)

    # --- Step 10: Generate new images ---
    print(f"\nGenerating {args.n_generate} new images...\n")
    generate_images(generator, noise_dim, args.n_generate)
