# Artistic-style-transfer

This repository contains the python code for artistic style transfer. In this case it transfers the Van Gogh's "Starry Night" to the landscape picture downloaded from the web source. It uses VGG19 pre-trained model for training. The model extracts style information from ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'] layers and content information from ['block5_conv2'] layers of VGG19 architecture. The model ooutputs both intermediate outputs from style and content layers.
# Architecture

[Input Image (224x224x3)]
       |
  [VGG19 Model]
       |
  ------------------------------------
 | Content Features from block5_conv2 |
 | Style Features from style layers   |
  ------------------------------------
       |
      [Loss Computation]
       |
  [Optimization (Style Transfer)]


# Run this repository.
Clone this repository
git clone git@github.com:Athish22/Artistic-style-transfer.git

edit the aug.sh and run the shell script.




