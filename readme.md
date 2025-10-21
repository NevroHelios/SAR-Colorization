# SAR Colorization using U-Net GAN

A deep learning project for colorizing Synthetic Aperture Radar (SAR) images using a U-Net based Generative Adversarial Network (GAN) architecture. This project converts grayscale SAR satellite imagery into pseudo-colored RGB images.

## Overview

Synthetic Aperture Radar (SAR) imagery provides valuable information about Earth's surface regardless of weather conditions or time of day. However, SAR images are inherently grayscale, making them difficult to interpret. This project leverages deep learning to automatically colorize SAR images, making them more intuitive for analysis and visualization.

The project implements a conditional GAN (cGAN) with a U-Net generator architecture that learns to map grayscale SAR images to colorized versions based on Sentinel-2 optical imagery.

## Architecture

### Generator (U-Net)
- **Encoder**: 7 downsampling blocks using Conv2d layers with LeakyReLU activation and BatchNorm
- **Bottleneck**: 512-channel feature representation
- **Decoder**: 7 upsampling blocks using ConvTranspose2d with ReLU activation and BatchNorm
- **Skip Connections**: Concatenation between encoder and decoder layers for preserving spatial information
- **Output**: 3-channel RGB image with Tanh activation

### Discriminator (PatchGAN)
- 4 convolutional blocks with BatchNorm and LeakyReLU
- Takes concatenated SAR + RGB images as input (4 channels)
- Outputs a patch-based classification map

### Loss Function
- Binary Cross Entropy with Logits Loss (BCEWithLogitsLoss)
- Adversarial loss for both generator and discriminator

## Features

- **U-Net Generator**: Preserves spatial information through skip connections
- **PatchGAN Discriminator**: Provides detailed feedback at the patch level
- **TensorBoard Integration**: Real-time monitoring of training metrics
- **Model Checkpointing**: Automatic saving of trained models
- **Prediction Pipeline**: Easy-to-use inference script for colorizing new SAR images

## Requirements

- Python 3.7+
- PyTorch
- torchsummary
- numpy
- scikit-image
- matplotlib
- tqdm
- tensorboard

## Installation

1. Clone the repository:
```bash
git clone https://github.com/NevroHelios/SAR-Colorization.git
cd SAR-Colorization
```

2. Create a virtual environment:
```bash
python -m venv sar_colorization
```

3. Activate the virtual environment:
- On Linux/Mac:
  ```bash
  source sar_colorization/bin/activate
  ```
- On Windows:
  ```bash
  .\sar_colorization\Scripts\activate
  ```

4. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

This project uses the ROIs2017 winter Sentinel-2 dataset. The dataset should be placed in a folder named `ROIs2017_winter_s2` in the project root directory.

Expected structure:
```
ROIs2017_winter_s2/
├── s2_1/
│   ├── ROIs2017_winter_s2_1_p1.tif
│   └── ...
├── s2_2/
└── ...
```

## Usage

### Training

Run the training script to train the GAN model:

```bash
python train.py
```

Training parameters can be modified in `train.py`:
- `epochs`: Number of training epochs (default: 30)
- `batch_size`: Batch size for training (default: 64)
- `lr`: Learning rate (default: 1e-3)

The trained generator model will be automatically saved to the `models/` directory.

### Monitoring Training

Monitor training progress in real-time using TensorBoard:

```bash
tensorboard --logdir=runs
```

Then open your browser and navigate to `http://localhost:6006/`

You can view:
- Generator loss over epochs
- Discriminator loss over epochs

### Prediction

To colorize a SAR image using a trained model:

```bash
python predict.py
```

Edit the `image_path` variable in `predict.py` to point to your SAR image. The script will display:
- Input grayscale SAR image
- Ground truth RGB image (if available)
- Predicted colorized image

## Project Structure

```
SAR-Colorization/
├── data.py           # Dataset and dataloader creation
├── data_viz.py       # Data visualization utilities
├── model.py          # Generator and Discriminator architectures
├── train.py          # Training script
├── predict.py        # Inference script
├── utils.py          # Helper functions for saving/loading models
├── requirements.txt  # Python dependencies
└── readme.md         # Project documentation
```

## Model Details

### Generator Model
- Input: 1-channel grayscale SAR image (1024x1024)
- Output: 3-channel RGB pseudo-color image (1024x1024)
- Architecture: U-Net with 7 encoder-decoder pairs
- Parameters: ~50M trainable parameters

### Discriminator Model
- Input: 4-channel concatenated image (SAR + RGB)
- Output: Patch-based real/fake classification
- Architecture: 4-layer convolutional network with BatchNorm

### Training Strategy
- Optimizer: Adam (β1=0.5, β2=0.999)
- Generator learning rate: 1e-4
- Discriminator learning rate: 1e-6
- Training alternates between discriminator and generator updates

## Results

The model learns to generate pseudo-colored SAR images that:
- Preserve spatial details from the original SAR imagery
- Generate realistic color distributions
- Maintain consistency across similar terrain types

## Alternative Model

The project also includes a simpler Linear Regression baseline model (`LR4ColSAR`) for comparison, which learns linear transformations from grayscale to RGB channels.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available for educational and research purposes.

## Acknowledgments

- Sentinel-2 data provided by ESA (European Space Agency)
- U-Net architecture inspired by Ronneberger et al., 2015
- cGAN architecture based on Isola et al., 2017 (Pix2Pix)

## Citation

If you use this project in your research, please cite:

```
@software{sar_colorization,
  title={SAR Colorization using U-Net GAN},
  author={NevroHelios},
  year={2024},
  url={https://github.com/NevroHelios/SAR-Colorization}
}
```

## Contact

For questions or issues, please open an issue on the GitHub repository.
