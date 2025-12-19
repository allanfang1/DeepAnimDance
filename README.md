## Introduction

The objective is to generate a video of a target person performing the same movements as a source person, using human pose estimation and neural networks. From: 

  - a source video (motion)
  - a target video (appearance)

We aim to synthesize new images of the target person that follow the motion of the source.

Following approaches are implemented:

  - Nearest Neighbor
  - Vanilla Neural Network: 26 Skeleton Input
  - Vanilla Neural Network: Skeleton Image Input
  - GAN



## Pose Extraction

Human pose estimation is performed using MediaPipe Pose. Each video frame is associated with a skeleton composed of either:

  - 26 values (reduced skeleton, 13 joints × 2D)
  - 99 values (full skeleton, 33 joints × 3D)

Skeletons are stored and reused for training and inference.



## Nearest Neighbor (GenNearest)

For a given input skeleton, the system searches for the closest skeleton (Euclidean distance) in the target video and returns the corresponding image.

Characteristics: 

  - No learning
  - Deterministic
  - Produces realistic images
  - Motion continuity is limited

Implementation:
- Input: Skeleton pose
- Method: Brute-force distance search
- Process: Compares input skeleton to all target video skeletons using skeleton.distance(), returns image with closest match
- Output: Retrieves corresponding frame from target video
- No training required

## Vanilla Neural Network (GenVanillaNN)

Two variants are implemented:

1 - Skeleton → Image (26D → Image)

Implementation:
- Input: Flattened skeleton vector (26-dim: 13 joints × 2D)
- Architecture: 5-layer ConvTranspose2d generator (26→128→64→32→8→3 channels)
- Upsampling: 1×1 → 4×4 → 8×8 → 16×16 → 32×32 → 64×64 images
- Training: MSE loss, Adam optimizer (lr=1e-3)
- Output: 64×64 RGB image normalized to [-1,1]

2 - Skeleton Image → Image (recommended)

Implementation:
- Input: 64×64 image with skeleton drawn on white background
- Architecture: Encoder-decoder (3→32→64→128 then 128→64→32→3 channels)
- Encoder: 3 Conv2d layers downsample 64×64 → 8×8
- Decoder: 3 ConvTranspose2d layers upsample 8×8 → 64×64
- BatchNorm + LeakyReLU between layers
- Training: Same as approach 2

Improvements made

  - Switched to skeleton-image input for better spatial consistency
  - Proper normalization to [-1, 1]
  - Stable training and inference



## Conditional GAN (GenGAN)

Architecture

Generator: CNN encoder–decoder (same as VanillaNN image-based)
Discriminator: Patch-based CNN
Input: skeleton image
Output: generated image of the target person
Loss function

The generator is trained with a combination of:

  - Adversarial loss (Binary Cross-Entropy)
  - L1 reconstruction loss with λ = 100.

Implementation:
- Generator: Same encoder-decoder as approach 3
- Discriminator: 5-layer CNN (3→32→64→128→256→1), outputs real/fake probability
- Loss: BCE loss + L1 loss (λ=100) for structure preservation
- Training: Adversarial training - D distinguishes real/fake, G fools D while minimizing L1 distance
- Input/Output: Same as approach 3

Observations

  - GAN output is visually less stable than VanillaNN
  - Sensitive to normalization and training duration
  - Produces sharper results when well trained



## How to Run the Demo

The project uses a Conda environment.
The provided tp_ml.yml file by Julie Digne can be used to recreate the environment (Remove linux dependencies if using Windows)

Model selection in DanceDemo.py

  - GEN_TYPE = 1 → Nearest Neighbor
  - GEN_TYPE = 2 → Vanilla Neural Network: 26 Skeleton Input
  - GEN_TYPE = 3 → Vanilla Neural Network: Skeleton Image Input
  - GEN_TYPE = 4 → GAN


## How to Train the Networks

To train Vanilla Neural Network: 26 Skeleton Input
  - set optSkeOrImage = 1 in main of GenVanillaNN.py
  - python GenVanillaNN.py
  - the trained model is saved in data/Dance/DanceGenVanillaFromSke26.pth

To train Vanilla Neural Network: Skeleton Image Input
  - set optSkeOrImage = 2 in main of GenVanillaNN.py
  - python GenVanillaNN.py
  - the trained model is saved in data/Dance/DanceGenVanillaFromSkeim.pth

To train GAN
  - python GenGAN.py
  - the trained model is saved in data/Dance/DanceGenGAN.pth
