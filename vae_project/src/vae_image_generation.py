"""
Convolutional Variational Autoencoder (Conv-VAE) with TensorFlow
for Image Generation Using Custom Image Folders

Usage:
    # Train and generate:
    python vae_image_generation.py --data_dir path/to/your/images
    python vae_image_generation.py --data_dir path/to/images --epochs 50 --batch_size 32

    # Generate from a saved model (no retraining):
    python generate_from_model.py --model_dir outputs/saved_model --n_generate 20

Steps:
    1. Import the Necessary Libraries
    2. Load Images from a Custom Folder
    3. Set Hyperparameters
    4. Define Convolutional Model Architecture (Encoder + Decoder)
    5. Define the Sampling Function
    6. Connect the Encoder and Decoder
    7. Define the Loss Function and Compile the Model
    8. Train the Model
    9. Save the Trained Model
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
def load_images_from_folder(folder_path, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, augment=True):
    """
    Load all images from a folder (recursively), convert to RGB, resize to
    (img_height x img_width), normalize to [0, 1], apply data augmentation,
    and split 80/20 into train/test sets.

    Augmentation (5x multiplier):
        - Original image
        - Horizontal flip
        - 90° rotation
        - 180° rotation
        - 270° rotation

    Args:
        folder_path: Path to folder containing image files.
        img_height:  Target image height (default: 128).
        img_width:   Target image width  (default: 128).
        augment:     If True, apply augmentation (flip + rotations) for ~5x data.

    Returns:
        x_train, x_test: NumPy arrays of shape (N, img_height, img_width, 3).
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
                img = Image.open(filepath).convert('RGB')        # RGB color
                img = img.resize((img_width, img_height))        # resize
                img_array = np.array(img, dtype=np.float32)
                images.append(img_array)
                original_count += 1

                # Data augmentation: flip + rotations = 5x total
                if augment:
                    images.append(np.fliplr(img_array))          # horizontal flip
                    images.append(np.rot90(img_array, k=1))      # 90°
                    images.append(np.rot90(img_array, k=2))      # 180°
                    images.append(np.rot90(img_array, k=3))      # 270°

            except Exception as e:
                print(f"  [SKIP] {filename}: {e}")
                skipped += 1

    if len(images) == 0:
        raise ValueError(
            f"No valid images found in '{folder_path}'.\n"
            f"Supported formats: {SUPPORTED_EXTENSIONS}"
        )

    data = np.array(images) / 255.0  # normalize to [0, 1]
    np.random.shuffle(data)

    # 80 / 20 train-test split
    split_idx = max(1, int(len(data) * 0.8))
    x_train = data[:split_idx]
    x_test = data[split_idx:] if split_idx < len(data) else data[:1]

    print(f"\nLoaded {original_count} original images from: {folder_path}")
    if augment:
        print(f"  Augmented (flip + rotations): {original_count} -> {len(data)} images")
    if skipped:
        print(f"  Skipped {skipped} unreadable files")
    print(f"  Image size : {img_height} x {img_width} x {IMG_CHANNELS} (RGB)")
    print(f"  x_train    : {x_train.shape}")
    print(f"  x_test     : {x_test.shape}\n")

    return x_train, x_test


# =============================================================================
# Step 3: Set Hyperparameters
# =============================================================================
learning_rate = 0.0005  # slightly lower for stable training with more data
latent_dim = 64  # 64-D latent space for richer RGB image encoding


# =============================================================================
# Step 4: Define Convolutional Model Architecture
# Conv2D encoder preserves spatial information (edges, textures, shapes, colors)
# Conv2DTranspose decoder upsamples back to full image resolution.
# Architecture: 128x128x3 -> 64x64x32 -> 32x32x64 -> 16x16x128 -> 8x8x256 -> latent -> reverse
# =============================================================================
def build_encoder(img_height, img_width, latent_dim):
    """Build a convolutional encoder: RGB image -> (z_mean, z_log_var)."""
    encoder_inputs = tf.keras.Input(shape=(img_height, img_width, IMG_CHANNELS))

    # Conv block 1: 128x128x3 -> 64x64x32
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(encoder_inputs)
    x = tf.keras.layers.BatchNormalization()(x)

    # Conv block 2: 64x64x32 -> 32x32x64
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Conv block 3: 32x32x64 -> 16x16x128
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Conv block 4: 16x16x128 -> 8x8x256
    x = tf.keras.layers.Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Flatten and project to latent space
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)

    z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)
    return encoder_inputs, z_mean, z_log_var


def build_decoder(img_height, img_width, latent_dim):
    """Build a convolutional decoder: latent vector -> reconstructed RGB image."""
    latent_inputs = tf.keras.Input(shape=(latent_dim,))

    # Project and reshape to 8x8x256 (matches encoder's last conv output)
    x = tf.keras.layers.Dense(8 * 8 * 256, activation='relu')(latent_inputs)
    x = tf.keras.layers.Reshape((8, 8, 256))(x)

    # Deconv block 1: 8x8x256 -> 16x16x256
    x = tf.keras.layers.Conv2DTranspose(256, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Deconv block 2: 16x16x256 -> 32x32x128
    x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Deconv block 3: 32x32x128 -> 64x64x64
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Deconv block 4: 64x64x64 -> 128x128x32
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Final conv to 3 RGB channels: 128x128x32 -> 128x128x3
    decoder_outputs = tf.keras.layers.Conv2D(IMG_CHANNELS, 3, padding='same', activation='sigmoid')(x)

    return latent_inputs, decoder_outputs


# =============================================================================
# Step 5: Define the Sampling Function
# Custom Keras layer for the reparameterization trick used in the VAE.
# =============================================================================
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# =============================================================================
# Step 6: Connect the Encoder and the Decoder
# Use keras functional API to connect the encoder and decoder parts.
# =============================================================================
def build_vae(img_height, img_width, latent_dim):
    """Assemble encoder + sampling + decoder into a full VAE."""
    encoder_inputs, z_mean, z_log_var = build_encoder(img_height, img_width, latent_dim)
    encoder_outputs = Sampling()([z_mean, z_log_var])
    encoder = tf.keras.Model(inputs=encoder_inputs,
                             outputs=[z_mean, z_log_var, encoder_outputs],
                             name='encoder')

    latent_inputs, decoder_outputs = build_decoder(img_height, img_width, latent_dim)
    decoder = tf.keras.Model(inputs=latent_inputs,
                             outputs=decoder_outputs,
                             name='decoder')
    return encoder, decoder


# =============================================================================
# Step 7: Define the Loss Function and Compile the Model
# The loss includes a reconstruction loss and a KL divergence loss.
# =============================================================================
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def compute_loss(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstructed = self.decoder(z)
        # Flatten both for binary crossentropy (handles the channel dim)
        x_flat = tf.reshape(x, [tf.shape(x)[0], -1])
        r_flat = tf.reshape(reconstructed, [tf.shape(reconstructed)[0], -1])
        reconstruction_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(x_flat, r_flat)
        )
        reconstruction_loss *= IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS  # scale per-pixel per-channel -> per-image
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        return reconstruction_loss + kl_loss

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            loss = self.compute_loss(data)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {'loss': loss}


# =============================================================================
# Step 8 & 9: Generate New Images
# =============================================================================
def generate_images(model, n_images, latent_dim, save_path="outputs/generated_images.png"):
    """Generate images by sampling random points in the latent space.

    Args:
        model:      Trained VAE model.
        n_images:   Number of images to generate.
        latent_dim: Dimensionality of the latent space.
        save_path:  File path to save the generated grid image.
    """
    random_latent_vectors = tf.random.normal(shape=(n_images, latent_dim))
    generated_images = model.decoder(random_latent_vectors).numpy()

    n_rows = int(np.ceil(n_images / 4))

    fig, axes = plt.subplots(n_rows, 4, figsize=(12, 3 * n_rows))
    axes = axes.flatten()
    for i in range(n_images):
        # RGB image: clip to [0,1] for clean display
        img = np.clip(generated_images[i], 0, 1)
        axes[i].imshow(img)
        axes[i].axis('off')
    # Hide any unused subplot slots
    for j in range(n_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    # Save to file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Generated images saved to: {os.path.abspath(save_path)}")

    # Also try to show interactively (non-blocking)
    plt.show(block=False)
    plt.close(fig)


# =============================================================================
# Main: parse arguments, load data, build model, train, and generate
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a VAE on your own images and generate new ones."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to the folder containing training images."
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs (default: 100)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Training batch size (default: 32)."
    )
    parser.add_argument(
        "--latent_dim", type=int, default=latent_dim,
        help=f"Latent space dimensionality (default: {latent_dim})."
    )
    parser.add_argument(
        "--n_generate", type=int, default=16,
        help="Number of images to generate after training (default: 16)."
    )
    parser.add_argument(
        "--model_dir", type=str, default="outputs/saved_model",
        help="Directory to save trained encoder/decoder (default: outputs/saved_model)."
    )
    parser.add_argument(
        "--no_augment", action="store_true",
        help="Disable data augmentation (default: augmentation is ON)."
    )
    args = parser.parse_args()

    # Override global latent_dim if user specified a different value
    latent_dim = args.latent_dim

    # --- Step 2: Load images ---
    x_train, x_test = load_images_from_folder(args.data_dir, augment=not args.no_augment)

    # --- Step 4-6: Build the VAE ---
    encoder, decoder = build_vae(IMG_HEIGHT, IMG_WIDTH, latent_dim)
    encoder.summary()
    decoder.summary()

    # --- Step 7: Compile ---
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    # --- Step 8: Train ---
    print(f"\nTraining for {args.epochs} epochs, batch_size={args.batch_size}\n")
    vae.fit(x_train, x_train, epochs=args.epochs, batch_size=args.batch_size)

    # --- Step 9: Save the trained model ---
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    encoder.save(os.path.join(model_dir, "encoder.keras"))
    decoder.save(os.path.join(model_dir, "decoder.keras"))
    # Save config so the generator script knows the settings
    config = {"img_height": IMG_HEIGHT, "img_width": IMG_WIDTH, "img_channels": IMG_CHANNELS, "latent_dim": latent_dim}
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nModel saved to: {os.path.abspath(model_dir)}")
    print(f"  - encoder.keras")
    print(f"  - decoder.keras")
    print(f"  - config.json")

    # --- Step 10: Generate ---
    print(f"\nGenerating {args.n_generate} new images...\n")
    generate_images(vae, args.n_generate, latent_dim)
