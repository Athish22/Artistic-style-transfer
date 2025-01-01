# Artistic Style Transfer
This repository provides Python code for artistic style transfer, allowing you to transfer the artistic style of Van Gogh's "Starry Night" (or any other chosen style image) onto a landscape picture or any other content image. The implementation leverages the VGG19 pre-trained model for extracting both style and content features.

# Overview
Artistic style transfer combines the style of one image with the content of another. This is achieved by:

Extracting style features from specific layers of the VGG19 model.
Extracting content features from a separate layer of the same model.
Iteratively updating a generated image to minimize the difference between its style features and the style target, as well as its content features and the content target.

# Model Architecture
Input
The input to the model is a 3-channel RGB image resized to 224x224 pixels.

Feature Extraction
Using the VGG19 pre-trained model, the following features are extracted:

Style Features: Extracted from these convolutional layers:
block1_conv1
block2_conv1
block3_conv1
block4_conv1
block5_conv1
Content Features:
Extracted from:
block5_conv2
Processing

The model outputs:
Intermediate style features from the specified style layers.
Intermediate content features from the content layer.

Optimization
The extracted features are used to compute two types of losses:

Style Loss: Measures the difference in the texture and patterns of the generated image and the style target using the Gram matrix.
Content Loss: Measures the difference in the structural content of the generated image and the content target.
These losses are optimized to iteratively update the generated image.

# Architecture Flow

[Input Image (224x224x3)]
       ↓
    [VGG19 Model]
       ↓
  ------------------------------------
 | Content Features from block5_conv2 |
 | Style Features from style layers   |
  ------------------------------------
       ↓
  [Loss Computation]
       ↓
[Optimization (Style Transfer)]

# Installation and Usage
Clone the Repository
To get started, clone this repository:

```bash
git clone git@github.com:Athish22/Artistic-style-transfer.git
cd Artistic-style-transfer
```

Prepare Input Images
Place your content image (e.g., a landscape picture) in the appropriate folder.
Place your style image (e.g., Van Gogh's "Starry Night") in the designated folder.
Edit the Shell Script
Customize the aug.sh script with your image paths, parameters, or any other configurations.

Run the Script
Execute the script to start the style transfer process:
```bash
sh aug.sh
```


# Results
The generated styled image will be saved in the output folder specified in the script. You can observe the transformation of the content image into a masterpiece inspired by the style of your chosen artwork.

# Dependencies
The following Python libraries are required:

- tensorflow
- numpy
- matplotlib
- Pillow (for image loading and processing)
- Install the dependencies using:

```bash
pip install tensorflow numpy matplotlib Pillow
```

# Credits
This implementation uses the VGG19 model pre-trained on the ImageNet dataset for feature extraction. The architecture is inspired by the seminal work on artistic style transfer.
