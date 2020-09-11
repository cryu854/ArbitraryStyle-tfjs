import tensorflow as tf

from model import vgg_preprocess
from utils import create_dir, load_image, imsave


class Inferencer:
    def __init__(self, model_path, result_path):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.result_path = result_path
        self.num = 0
        create_dir(self.result_path)
        

    @tf.function(experimental_relax_shapes=True)
    def __call__(self, content_image, style_image, alpha):
        alpha = tf.expand_dims(tf.cast(alpha, tf.float32), axis=0)
        _, _, stylized = self.model([content_image, style_image, alpha])
        return stylized

 
    def preprocess_file(self, image_file):
        image = load_image(image_file, training=False)
        return vgg_preprocess(image)


    def save(self, image):
        self.num += 1
        imsave(image, f'{self.result_path}/result{self.num}.jpg')