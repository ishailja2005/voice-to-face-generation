# Speech-to-Face Generation using PyTorch

## Quick start

Run all commands from the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

First train the face autoencoder. This creates the decoder checkpoint required by the speech-to-face model:

```powershell
python training\train_face_autoencoder.py --epochs 200
```

Then train the speech-to-face model:

```powershell
python training\train.py --epochs 200
```

Generate an image using one of the processed voice embeddings:

```powershell
python inference\generate.py
```

The image is saved to `outputs/generated_face.png`. You can choose a different `.npy` embedding with `--embedding path\to\file.npy`.

### Optional: clearer face decoder training

The raw VGGFace images vary in pose and lighting, so align them before using them as extra training data:

```powershell
python preprocessing\align_faces.py --max-faces-per-identity 50
python training\train_face_autoencoder.py --epochs 50 --aligned-face-root data\aligned_faces
python training\train.py --epochs 200
python inference\generate.py
```

The project has only 55 paired voice samples. The face-autoencoder stage uses the processed, aligned face images by default; this keeps the output frontal and reduces blur. The voice-to-face stage uses the paired voice embeddings. Raw VGGFace images can be enabled with `--use-raw-faces`, but they have varied pose and lighting and are expected to blur output without a separate face-alignment step. The model can learn the available examples, but it cannot produce an exact or reliably identifiable face for new voices. The saved checkpoints are the best ones observed during training, and the learning rate is automatically reduced when loss stops improving.

## Project Overview

This project aims to generate a human face from a speech embedding using Deep Learning. The system takes a preprocessed voice embedding as input and predicts the corresponding face image using a Generator Network implemented in PyTorch.

The complete pipeline consists of dataset preparation, preprocessing, custom dataset creation, model development, training, inference, and a user interface for prediction.

---

# Project Workflow

```
Environment Setup
        ↓
Dataset Preparation
        ↓
Data Preprocessing
        ↓
Custom PyTorch Dataset
        ↓
DataLoader
        ↓
Generator Network
        ↓
Loss Function
        ↓
Training Loop
        ↓
Inference
        ↓
User Interface
```

---

# Project Directory Structure

```
speech_to_face/
│
├── checkpoints/
│   └── generator.pth
│
├── data/
│   └── processed/
│       ├── embeddings/
│       ├── faces/
│       ├── manifest.csv
│       └── bin/
│
├── dataset/
│   └── dataset.py
│
├── inference/
│   └── generate.py
│
├── models/
│   └── generator.py
│
├── outputs/
│   └── generated_face.png
│
├── preprocessing/
│
├── training/
│   ├── dataloader.py
│   ├── loss.py
│   └── train.py
│
├── test_dataset.py
├── test_dataloader.py
├── test_generator.py
├── test_loss.py
│
├── requirements.txt
└── README.md
```

---

# Step 1 : Environment Setup

## Objective

Prepare the Python environment and install all required libraries.

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Windows

```bash
.venv\Scripts\activate
```

Linux/Mac

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install torch torchvision numpy pandas pillow matplotlib
```

---

# Step 2 : Dataset Preparation

## Objective

Prepare the processed dataset that will be used for training.

The processed dataset contains

- Voice embeddings (.npy)
- Face images (.jpg)
- Manifest file (CSV)

### Folder Structure

```
data/
    processed/
        embeddings/
            id00040/
                sample.npy

        faces/
            xxxx_aligned.jpg

        manifest.csv
```

The preprocessing has already been completed and therefore the raw datasets are not required to train the current model.

---

# Step 3 : Data Preprocessing

## Objective

Convert raw voice recordings into numerical embeddings and align corresponding face images.

Outputs

- Voice Embeddings (.npy)
- Aligned Face Images
- Manifest CSV

The preprocessing module is executed only once while preparing the dataset.

---

# Step 4 : Custom Dataset

File

```
dataset/dataset.py
```

## Purpose

Implements a custom PyTorch Dataset class.

Responsibilities

- Reads manifest.csv
- Loads voice embeddings
- Loads corresponding face images
- Applies image transformations
- Converts data into tensors
- Returns

```
{
    "embedding": embedding,
    "image": image
}
```

Functions implemented

```
__init__()
__len__()
__getitem__()
```

---

# Dataset Testing

File

```
test_dataset.py
```

Purpose

Verifies

- Dataset loads successfully
- Total samples
- Embeddings are read correctly
- Images are loaded correctly

Expected Output

```
Dataset loaded successfully

Total samples : 55
```

---

# Step 5 : DataLoader

File

```
training/dataloader.py
```

## Purpose

Creates mini-batches for efficient training.

Responsibilities

- Shuffle dataset
- Batch data
- Feed batches to the model

Typical batch size

```
8
```

Each batch contains

```
Embeddings

Shape

(8,256)

Images

Shape

(8,3,160,160)
```

---

# DataLoader Testing

File

```
test_dataloader.py
```

Expected Output

```
torch.Size([8,256])

torch.Size([8,3,160,160])
```

---

# Step 6 : Generator Network

File

```
models/generator.py
```

## Purpose

Generate a face image from a speech embedding.

Architecture

```
Voice Embedding (256)

↓

Fully Connected Layer

↓

ReLU

↓

Fully Connected Layer

↓

Reshape

↓

512 × 5 × 5 Feature Map

↓

ConvTranspose2D

↓

BatchNorm

↓

ReLU

↓

ConvTranspose2D

↓

BatchNorm

↓

ReLU

↓

ConvTranspose2D

↓

BatchNorm

↓

ReLU

↓

ConvTranspose2D

↓

BatchNorm

↓

ReLU

↓

ConvTranspose2D

↓

Tanh

↓

Generated Face

(3 × 160 × 160)
```

---

# Generator Testing

File

```
test_generator.py
```

Expected Output

```
torch.Size([8,3,160,160])
```

---

# Step 7 : Loss Function

File

```
training/loss.py
```

Purpose

Measures the difference between

- Generated face
- Ground truth face

Loss Used

```
Mean Squared Error (MSE)
```

Implementation

```python
criterion = nn.MSELoss()
```

---

# Loss Testing

File

```
test_loss.py
```

Expected Output

```
tensor(...)
```

A single scalar value indicating the reconstruction error.

---

# Step 8 : Training

File

```
training/train.py
```

## Purpose

Train the Generator Network.

Training Pipeline

```
Load Batch

↓

Forward Pass

↓

Generate Face

↓

Calculate Loss

↓

Backward Pass

↓

Optimizer Step

↓

Update Weights
```

Optimizer

```
Adam
```

Learning Rate

```
0.0002
```

Epochs

```
20
```

Output

```
Epoch 1

Loss

...

Epoch 20

Loss

...
```

After training

```
generator.pth
```

is saved inside

```
checkpoints/
```

---

# Step 9 : Inference

File

```
inference/generate.py
```

Purpose

Generate a face using a trained model.

Pipeline

```
Load Model

↓

Load Voice Embedding

↓

Generate Face

↓

Convert Tensor to Image

↓

Save Image
```

Output

```
outputs/generated_face.png
```

---

# Step 10 : User Interface

This module will provide an interface where users can

- Upload speech
- Generate embedding
- Load trained model
- Generate face
- Display generated face

Possible frameworks

- Tkinter
- Streamlit
- Flask

---

# Testing Summary

| File | Purpose |
|------|----------|
| test_dataset.py | Test Dataset |
| test_dataloader.py | Test DataLoader |
| test_generator.py | Test Generator |
| test_loss.py | Test Loss Function |

Each component should be verified before moving to the next stage.

---

# Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Pillow
- TorchVision
- Matplotlib

---

# Future Improvements

- Train on a significantly larger dataset.
- Increase the number of training epochs.
- Replace MSE with L1 or Perceptual Loss.
- Introduce a Discriminator to build a GAN-based architecture.
- Add Residual Blocks and Attention mechanisms.
- Support direct speech (.wav) input instead of precomputed embeddings.
- Deploy the model as a web application using Streamlit or Flask.

---

# Conclusion

This project implements a complete Speech-to-Face Generation pipeline using PyTorch. It covers every stage of the workflow, from dataset preparation and preprocessing to model training and inference. The current implementation provides a baseline generator capable of producing face images from voice embeddings and serves as a foundation for future improvements in image quality, model architecture, and deployment.# voice-to-face-generation
