import os
import numpy as np
import tensorflow as tf
from PIL import Image
from model import vgg_preprocess


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f'Directory {dir} createrd')
    else:
        print(f'Directory {dir} already exists')  

    return dir


def load_image(file_name, training=True):
    """Load an image from directory
    Args:
        image : Path to the image
        training : True for argumentation
    Return:
        image = [H, W, C] for training=True, 
                [1, H, W, C] for training=False,
                with scale between [0, 255] tf.float32
    """
    image = tf.io.read_file(file_name)
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = image[tf.newaxis, ...]

    if training:
        image = tf.squeeze(image, axis=0)
        min_dim = 512
        shape = tf.cast(tf.shape(image)[:-1], tf.float32)
        short_dim = tf.reduce_min(shape)
        scale = min_dim / short_dim
        new_shape = tf.cast(shape * scale, tf.int32)
        image = tf.image.resize(image, new_shape,
                                method=tf.image.ResizeMethod.BILINEAR)
        image = tf.image.random_crop(image, [256, 256, 3])

    return image


def imsave(image, path):
    """Save an image to the path
    Args:
        image : tf.float32 with scale between [0., 255.]
        path : Path to save the image
    """
    image = clip_0_255(image)
    image = Image.fromarray(np.array(image).astype(np.uint8).squeeze())
    image.save(path)


def clip_0_255(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)