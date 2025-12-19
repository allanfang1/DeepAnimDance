Introduction

The objective is to generate a video of a target person performing the same movements as a source person, using human pose estimation and neural networks. From: 

  - a source video (motion)
  - a target video (appearance)

We aim to synthesize new images of the target person that follow the motion of the source.

Following methods are implemented:

  - Nearest Neighbor
  - Vanilla Neural Network
  - GAN



Pose Extraction

Human pose estimation is performed using MediaPipe Pose. Each video frame is associated with a skeleton composed of either:

  - 26 values (reduced skeleton, 13 joints × 2D)
  - 99 values (full skeleton, 33 joints × 3D)

Skeletons are stored and reused for training and inference.



Nearest Neighbor (GenNearest)

For a given input skeleton, the system searches for the closest skeleton (Euclidean distance) in the target video and returns the corresponding image.

Characteristics: 

  - No learning
  - Deterministic
  - Produces realistic images
  - Motion continuity is limited

This method serves as a baseline and was not modified.


Vanilla Neural Network (GenVanillaNN)

Two variants are implemented:

1 - Skeleton → Image (26D → Image)

Input: reduced skeleton (26 values)
Architecture: transposed convolutions
Loss: Mean Squared Error (MSE)

2 - Skeleton Image → Image (recommended)

Input: an image with the skeleton drawn on it
Architecture: encoder–decoder CNN
Output resolution: 64×64
Loss: Mean Squared Error (MSE)

Improvements made

  - Switched to skeleton-image input for better spatial consistency
  - Proper normalization to [-1, 1]
  - Stable training and inference



Conditional GAN (GenGAN)

Architecture

Generator: CNN encoder–decoder (same as VanillaNN image-based)
Discriminator: Patch-based CNN
Input: skeleton image
Output: generated image of the target person
Loss function

The generator is trained with a combination of:

  - Adversarial loss (Binary Cross-Entropy)
  - L1 reconstruction loss with λ = 100.

Observations

  - GAN output is visually less stable than VanillaNN
  - Sensitive to normalization and training duration
  - Produces sharper results when well trained



How to Run the Demo

The project uses a Conda environment.
The provided tp_ml.yml file by Julie Digne can be used to recreate the environment (Remove linux dependencies if using Windows )

Model selection in DanceDemo.py

  - GEN_TYPE = 1 → Nearest Neighbor
  - GEN_TYPE = 3 → Vanilla NN (skeleton image)
  - GEN_TYPE = 4 → GAN


How to Train the Networks

To train Vanilla Neural Network
  - python GenVanillaNN.py
  - the trained model is saved in data/Dance/DanceGenVanillaFromSkeim.pth

To train GAN
  - python GenGAN.py
  - the trained model is saved in data/Dance/DanceGenGAN.pth