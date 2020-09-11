import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Conv2D, MaxPooling2D


def adaptive_instance_norm(content_feature, style_feature, epsilon=1e-5):
    c_mean, c_var = tf.nn.moments(content_feature, axes=[1, 2], keepdims=True)
    s_mean, s_var = tf.nn.moments(style_feature, axes=[1, 2], keepdims=True)
    c_std = tf.sqrt(c_var + epsilon)
    s_std = tf.sqrt(s_var + epsilon)

    return s_std * ((content_feature - c_mean) / c_std) + s_mean


class ReflectConv2D(Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='valid', activation=None, **kwargs):
        super(ReflectConv2D, self).__init__(**kwargs)
        pad = kernel_size // 2
        self.relu = activation
        self.paddings = tf.constant([[0, 0], [pad, pad],[pad, pad], [0, 0]])
        self.conv2D = Conv2D(filters=filters,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding='valid')

    def call(self, inputs):
        x = tf.pad(inputs, self.paddings, mode='REFLECT')
        x = self.conv2D(x)
        if self.relu:
            x = tf.nn.relu(x)

        return x

    def get_config(self):
        config = super(ReflectConv2D, self).get_config()

        return config
        
        
class ResizeSampling(Layer):
    def __init__(self, stride, **kwargs):
        super(ResizeSampling, self).__init__(**kwargs)
        self.stride = stride

    def call(self, inputs):
        new_h = tf.shape(inputs)[1] * self.stride
        new_w = tf.shape(inputs)[2] * self.stride
        output = tf.image.resize(inputs, [new_h, new_w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return output
       
    def get_config(self):
        config = super(ResizeSampling, self).get_config()
        config.update({"stride":self.stride})

        return config


def Decoder(reflect_padding, name='Decoder'):
    conv = ReflectConv2D if reflect_padding else Conv2D
    x_in = Input(shape=(None, None, 512))
    x = conv(256, 3, 1, padding='same', activation='relu', name='conv1_1')(x_in)
    x = ResizeSampling(2, name='up1')(x)
    x = conv(256, 3, 1, padding='same', activation='relu', name='conv2_1')(x)
    x = conv(256, 3, 1, padding='same', activation='relu', name='conv2_2')(x)
    x = conv(256, 3, 1, padding='same', activation='relu', name='conv2_3')(x)
    x = conv(128, 3, 1, padding='same', activation='relu', name='conv2_4')(x)
    x = ResizeSampling(2, name='up2')(x)
    x = conv(128, 3, 1, padding='same', activation='relu', name='conv3_1')(x)
    x = conv(64, 3, 1, padding='same', activation='relu', name='conv3_2')(x)
    x = ResizeSampling(2, name='up3')(x)
    x = conv(64, 3, 1, padding='same', activation='relu', name='conv4_1')(x)
    x_out = conv(3, 3, 1, padding='same', name='conv4_2')(x) 
 
    return Model(inputs=[x_in], outputs=[x_out], name=name)


def Encoder(reflect_padding, weights='imagenet', name='Encoder'):
    EXTRACT_LAYERS = ['block1_conv1',
                      'block2_conv1',
                      'block3_conv1',
                      'block4_conv1']
    vgg = VGG19(reflect_padding, weights=weights)
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in EXTRACT_LAYERS]

    return Model(inputs=[vgg.input], outputs=outputs, name=name)


def vgg_preprocess(inputs):
    VGG_MEAN = tf.constant([103.939, 116.779, 123.68])
    bgr_inputs = tf.reverse(inputs, axis=[-1])
    preprocessed_inputs = tf.subtract(bgr_inputs, VGG_MEAN)

    return preprocessed_inputs


def VGG19(reflect_padding, weights='imagenet', name='vgg19'):
    conv = ReflectConv2D if reflect_padding else Conv2D
    inputs = Input(shape=(None, None, 3))
    # Block 1
    x = conv(64, 3, activation='relu', padding='same', name='block1_conv1')(inputs)
    x = conv(64, 3, activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    # Block 2
    x = conv(128, 3, activation='relu', padding='same', name='block2_conv1')(x)
    x = conv(128, 3, activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    # Block 3
    x = conv(256, 3, activation='relu', padding='same', name='block3_conv1')(x)
    x = conv(256, 3, activation='relu', padding='same', name='block3_conv2')(x)
    x = conv(256, 3, activation='relu', padding='same', name='block3_conv3')(x)
    x = conv(256, 3, activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    # Block 4
    x = conv(512, 3, activation='relu', padding='same', name='block4_conv1')(x)
    x = conv(512, 3, activation='relu', padding='same', name='block4_conv2')(x)
    x = conv(512, 3, activation='relu', padding='same', name='block4_conv3')(x)
    x = conv(512, 3, activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    # Block 5
    x = conv(512, 3, activation='relu', padding='same', name='block5_conv1')(x)
    x = conv(512, 3, activation='relu', padding='same', name='block5_conv2')(x)
    x = conv(512, 3, activation='relu', padding='same', name='block5_conv3')(x)
    x = conv(512, 3, activation='relu', padding='same', name='block5_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    model = Model(inputs=inputs, outputs=x, name=name)


    """ Load VGG19's pre-trained weights """
    WEIGHTS_PATH_NO_TOP = ('https://storage.googleapis.com/tensorflow/'
                           'keras-applications/vgg19/'
                           'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

    if weights == 'imagenet' or weights is None:
        weights_path = tf.keras.utils.get_file(
            'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
            WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='253f8cb515780f3b799900260a226db6')
        model.load_weights(weights_path)
    elif os.path.exists(weights):
        model.load_weights(weights)
    else:
        raise ValueError('The `weights` argument should be either '
                         '`None` , `imagenet`(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    return model


def get_STN(Encoder, Decoder, name='Style_Transfer_Network'):
    content_input = Input(shape=(None, None, 3))
    style_input = Input(shape=(None, None, 3))
    alpha = Input(shape=(1,))

    content_features = Encoder(content_input)
    style_features = Encoder(style_input)
    latent_code = adaptive_instance_norm(content_features[-1], style_features[-1])
    output = Decoder((1-alpha) * content_features[-1] + alpha * latent_code)

    return Model(inputs=[content_input, style_input, alpha], outputs=[style_features, latent_code, output], name=name)