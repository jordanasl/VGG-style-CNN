#VGG-style CNN with an MLP classification head

The proposed model is a deep convolutional neural network (CNN) designed for image classification on the CIFAR-10 dataset. It follows a VGG-style architecture, characterized by the use of stacked small (3×3) convolutional kernels, progressive channel expansion, and spatial downsampling via max-pooling.

Overall architecture: VGG-style Convolutional Neural Network with an MLP classification head
Input: 32×32 RGB images (3 channels)

Convolutional feature extractor (CNN backbone):
•8 convolutional layers
•3×3 kernels with padding = 1
•Channel progression: 3 → 32 → 64 → 128 → 256
•Batch Normalization after each convolution
•ReLU activation after each BatchNorm
•4 convolutional blocks, each followed by:
•2×2 Max Pooling for spatial downsampling

Flattening layer: Converts final feature maps into a 1D feature vector

Classification head (MLP):
•3 fully connected layers
•Layer sizes: 256 → 128 → 10
•ReLU activations in hidden layers
•Dropout (p = 0.5) between fully connected layers for regularization
•Output layer:
•10 output logits corresponding to the CIFAR-10 classes

Training details:
•Loss function: Cross-entropy loss
•Optimizer: Adam with weight decay (L2 regularization)
•Learning rate scheduling: Cosine Annealing

Model characteristics:
•Feed-forward, sequential architecture
•No residual or skip connections
•Designed for small-scale image classification tasks (e.g., CIFAR-10)
