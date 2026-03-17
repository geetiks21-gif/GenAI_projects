# VAE & GAN Image Generator

Generative AI image models built with TensorFlow — train on your own images and generate new ones.

- **VAE** (Variational Autoencoder): Learns a compressed representation and generates new images by sampling from it.
- **GAN** (Generative Adversarial Network): A Generator and Discriminator compete against each other to produce realistic images.

## Setup

```bash
cd D:\Python_GenAI_Projects\vae_project
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
```

## Usage

Drop your training images (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp`) into `sample_images/` or any folder.

### VAE

```bash
python src\vae_image_generation.py --data_dir sample_images

# With custom settings:
python src\vae_image_generation.py --data_dir sample_images --epochs 50 --batch_size 32 --latent_dim 32

# Generate from saved VAE model (no retraining):
python src\generate_from_model.py --model_dir outputs/saved_model --n_generate 20
```

### GAN

```bash
python src\gan_image_generation.py --data_dir sample_images

# With custom settings:
python src\gan_image_generation.py --data_dir sample_images --epochs 200 --batch_size 32 --noise_dim 128

# Generate from saved GAN model (no retraining):
python src\generate_from_gan.py --model_dir outputs/gan_saved_model --n_generate 20
```

## CLI Arguments

### VAE Arguments

| Argument       | Default      | Description                        |
|----------------|--------------|------------------------------------|
| `--data_dir`   | *(required)* | Path to folder with training images|
| `--epochs`     | 100          | Number of training epochs          |
| `--batch_size` | 32           | Training batch size                |
| `--latent_dim` | 64           | Latent space dimensions            |
| `--n_generate` | 16           | Number of images to generate       |

### GAN Arguments

| Argument             | Default      | Description                              |
|----------------------|--------------|------------------------------------------|
| `--data_dir`         | *(required)* | Path to folder with training images      |
| `--epochs`           | 200          | Number of training epochs                |
| `--batch_size`       | 32           | Training batch size                      |
| `--noise_dim`        | 128          | Noise vector dimensionality              |
| `--n_generate`       | 16           | Number of images to generate             |
| `--preview_interval` | 10           | Save progress images every N epochs      |

## Project Structure

```
vae_project/
├── src/
│   ├── vae_image_generation.py   # VAE training & generation
│   ├── generate_from_model.py    # Generate from saved VAE
│   ├── gan_image_generation.py   # GAN training & generation
│   └── generate_from_gan.py      # Generate from saved GAN
├── sample_images/                # Drop your training images here
├── outputs/                      # Generated outputs
│   ├── saved_model/              # Saved VAE encoder/decoder
│   ├── gan_saved_model/          # Saved GAN generator/discriminator
│   └── gan_progress/             # GAN training progress snapshots
├── requirements.txt
└── README.md
```
