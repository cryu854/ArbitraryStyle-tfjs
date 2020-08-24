import tensorflow as tf
import time

from model import vgg_preprocess
from utils import create_dir, load_image, imsave


class Inferencer:
    def __init__(self, model_dir, result_dir):
        self.model = tf.keras.models.load_model(model_dir, compile=False)
        self.result_dir = result_dir
        create_dir(self.result_dir)


    def __call__(self, content_file, style_file, alpha):
        start = time.perf_counter()

        content_image = load_image(content_file, training=False)
        style_image = load_image(style_file, training=False)
        content_image = vgg_preprocess(content_image)
        style_image = vgg_preprocess(style_image)
        alpha = tf.cast(alpha, tf.float32)

        _, _, stylized = self.model([content_image, style_image, alpha])

        print(f'Time taken is {time.perf_counter()-start:.2f} sec')
        imsave(stylized, f'{self.result_dir}/result.jpg')