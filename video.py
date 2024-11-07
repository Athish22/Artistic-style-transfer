import tensorflow_hub as hub
import os
import tensorflow as tf
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from tensorflow.keras.preprocessing.image import load_img, img_to_array


os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False


content_path = "/cig/common04nb/students/deaallay/Waste/big/frames"
style_path = "/cig/common04nb/students/deaallay/Waste/big/van.jpg"

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def load_and_process_image(img_paths):
    preprocessed_img = []
    if isinstance(img_paths, str) and os.path.isfile(img_paths):
        img_paths = [img_paths] 
    elif os.path.isdir(img_paths):
        img_paths = [os.path.join(img_paths, fname) for fname in os.listdir(img_paths)
                     if fname.endswith(('.png', '.jpg', '.jpeg'))]

    for img_path in img_paths:
        img = load_img(img_path, target_size=(256, 256))  # Change to 256x256 for the model
        img_array = img_to_array(img)
        img_array = tf.convert_to_tensor(img_array)
        img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension
        preprocessed_img.append(img_array)

    return preprocessed_img

def load_image_tensor(path_to_img):  # Renamed this function
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (256, 256))  # Resize to model input size
    img = img[tf.newaxis, :]  # Add batch dimension
    return img

def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


content_images = load_and_process_image(content_path)
style_image = load_image_tensor(style_path)


plt.subplot(1, 2, 1)
imshow(content_images[0], 'Content Image')  # Display the first content image

plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')
plt.show()


hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')


stylized_images = []
for content_image in content_images:
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    stylized_images.append(tensor_to_image(stylized_image))


for i, stylized_image in enumerate(stylized_images):
    plt.subplot(1, len(stylized_images), i + 1)
    plt.imshow(stylized_image)
    plt.title(f'Stylized Image {i + 1}')
plt.imsave('style_image {i}.png', stylized_image)
plt.show()
