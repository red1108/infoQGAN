# InfoQGAN

InfoQGAN is the quantum version of InfoGAN. This repository contains code for training and testing the InfoQGAN model compare to the QGAN, InfoGAN, GAN.

## Getting Started

You will need to install the appropriate packages via requirements.txt.

Before getting started, create a `runs` folder in the root directory. This folder is used to store TensorBoard data and training logs, and it is ignored by git.

The model is small enough to run without a GPU.

## Directory Structure

```bash
INFOQGAN/
├── data/                           # Folder for storing data-related files
│   ├── 2D/                         # Data used for 2D training
│   ├── IRIS/                       # Data used for IRIS data augmentation
├── modules/                        # Folder for model components and utility code
│   ├── Discriminator.py            # Code for the Discriminator model
│   ├── MINE.py                     # Code for the Mutual Information Neural Estimator (MINE)
│   ├── Generator.py           # Code for generating 2D distributions
│   ├── QGenerator.py                     # Code defining the QGAN model
│   └── utils.py                    # Collection of utility functions
├── runs/                           # Folder for storing TensorBoard data and training logs
├── savepoints/                     # Folder for saving autoencoder parameters
├── visualize/                      # Folder for image generation and visualization code
│   ├── tmp/                        # Temporary folder for images generated during training (ignored by Git)
│   ├── disentanglement.ipynb 
│   ├── mode_collapse_box.ipynb 
│   ├── mode_collapse_diamond_timeline.ipynb
├── .gitignore
├── mnist_train.py                  # (Important) Python file for executing MNIST training
├── mnist_train.ipynb               # Notebook for executing MNIST training
├── 2D_prepare.ipynb                # Notebook for generating data for 2D experiments
├── 2D_train.ipynb                  # Notebook for training 2D experiments
├── 2D_run.ipynb                    # Notebook for loading the trained 2D model and generating distributions
├── README.md                       # File containing project description and usage instructions
├── requirements.txt                # Written for Python 3.9.
```

### Requirements

- Python 3.9 or higher (**This code is written based on Python 3.9**)
- The required packages are defined in the `requirements.txt` file.

To install the dependencies, run the following command:
```bash
pip install -r requirements.txt
```

### How to Run
If you want to train with InfoQGAN:
```bash
python mnist_train.py --model_type InfoQGAN --DIGITS_STR 0123456789 --DIGIT 1 --G_lr 0.01 --M_lr 0.0001 --D_lr 0.001 --coeff 0.05 --epochs 300 --latent_dim 16 --num_images_per_class 2000
```

If you want to train with QGAN:
```bash
python mnist_train.py --model_type QGAN --DIGITS_STR 0123456789 --DIGIT 1 --G_lr 0.01 --M_lr 0.0001 --D_lr 0.001 --coeff 0.05 --epochs 300 --latent_dim 16 --num_images_per_class 2000
```