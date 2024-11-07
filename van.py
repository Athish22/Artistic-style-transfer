import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
import sys
import os

"""def load_and_process_image(img_path):
    img = load_img(img_path, target_size=(1024, 1024))  
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)  
    return img
"""
def load_and_process_image(img_paths):
    preprocessed_img = []
    if os.path.isdir(img_paths):
        img_paths = [os.path.join(img_paths, fname) for fname in os.listdir(img_paths)
                     if fname.endswith(('.png', '.jpg', '.jpeg'))]  # Filtering image files

    for img_path in img_paths:
        img = load_img(img_path, target_size=(224, 224))  # Resizing to VGG19 input size
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # Preprocessing for VGG19

        img_array = img_to_array(img).astype(np.float32)
        preprocessed_img.append(img_array)

    return preprocessed_img
def deprocess_image(processed_img):
    x = processed_img.copy()
    x = x.reshape((x.shape[1], x.shape[2], 3))  
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]  
    x = np.clip(x, 0, 255).astype('uint8')
    return x
content_image = "/cig/common04nb/students/deaallay/Waste/big/frames"
style_image = "/cig/common04nb/students/deaallay/Waste/big/vang"
output_folder = "/cig/common04nb/students/deaallay/Waste"
content_image = load_and_process_image(content_image)
style_image = load_and_process_image(style_image)


content_layers = ['block5_conv2'] 
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']


def build_model(style_layers, content_layers):
    vgg = vgg19.VGG19(weights='imagenet', include_top=False, input_shape=(1024, 1024, 3))
    vgg.trainable = False
    

    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    
    model_outputs = style_outputs + content_outputs
    
    return Model(inputs=vgg.input, outputs=model_outputs)


def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

def style_loss(style_output, style_target):
    S = gram_matrix(style_output)
    T = gram_matrix(style_target)
    return tf.reduce_mean(tf.square(S - T))


def content_loss(content_output, content_target):
    return tf.reduce_mean(tf.square(content_output - content_target))

def compute_loss(model, loss_weights, init_image, style_targets, content_targets):
    style_weight, content_weight = loss_weights
    
    model_outputs = model(init_image)
    
    style_output_features = model_outputs[:len(style_layers)]
    content_output_features = model_outputs[len(style_layers):]
    
    style_score = 0
    content_score = 0

    for target_style, comb_style in zip(style_targets, style_output_features):
        style_score += style_loss(comb_style, target_style)
        
    for target_content, comb_content in zip(content_targets, content_output_features):
        content_score += content_loss(comb_content, target_content)
    
    style_score *= style_weight
    content_score *= content_weight
    
    loss = style_score + content_score
    return loss


def compute_grads(cfg):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    total_loss = all_loss
    return tape.gradient(total_loss, cfg['init_image']), total_loss


model = build_model(style_layers, content_layers)
style_targets = model(style_image)[:len(style_layers)]
content_targets = model(content_image)[len(style_layers):]


init_image = tf.Variable(content_image, dtype=tf.float32)
optimizer = tf.optimizers.Adam(learning_rate=0.02)


style_weight = 1e-2
content_weight = 1e4
loss_weights = (style_weight, content_weight)


iterations = 100000
for i in range(1, iterations + 1):
    grads, all_loss = compute_grads({
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'style_targets': style_targets,
        'content_targets': content_targets
    })
    optimizer.apply_gradients([(grads, init_image)])
    if i % 1000 == 0:
        print(f"Iteration {i}, Loss: {all_loss}")
        output_file_path = f"{output_folder}/output_image_{i}.png"  
        plt.imshow(deprocess_image(init_image.numpy()))  
        plt.title('Styled Image')  
        plt.axis('off')  
        plt.savefig(output_file_path)  
        plt.show()  


