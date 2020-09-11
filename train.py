import tensorflow as tf
import numpy as np
import time
import os

from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import InverseTimeDecay
from tensorflow.keras.losses import MeanSquaredError

from model import Decoder, Encoder, get_STN, vgg_preprocess
from utils import create_dir, load_image, imsave


class Trainer:
    def __init__(self,
                 content_path,
                 style_path,
                 batch_size,
                 model_path,
                 debug,
                 validate_content,
                 validate_style,
                 style_weight,
                 content_weight,
                 reflect_padding,
                 num_epochs,
                 learning_rate,
                 lr_decay):

        self.style_weight = style_weight
        self.content_weight = content_weight
        self.num_epochs = num_epochs
        self.learning_schedule = InverseTimeDecay(initial_learning_rate=learning_rate, decay_steps=1, decay_rate=lr_decay)
        self.optimizer = Adam(learning_rate=self.learning_schedule, beta_1=0.9)
        self.mse = MeanSquaredError()
        self.model_path = model_path
        self.debug = debug
        self.build_model_ckpt(reflect_padding)
        self.content_dataset = self.create_dataset(content_path, batch_size)
        self.style_dataset = self.create_dataset(style_path, batch_size)
        
        if debug:
            self.create_summary_writer()
            self.create_metrics()
            self.validate_content = load_image(validate_content, training=False)
            self.validate_style = load_image(validate_style, training=False)
            self.validate_content = vgg_preprocess(self.validate_content)
            self.validate_style = vgg_preprocess(self.validate_style)
            create_dir('./results')
            

    def build_model_ckpt(self, reflect_padding):
        create_dir(self.model_path)
        create_dir('./checkpoints')
        self.decoder = Decoder(reflect_padding)
        self.encoder = Encoder(reflect_padding)
        self.stn = get_STN(self.encoder, self.decoder)
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(0),
                                        Decoder=self.decoder,
                                        Encoder=self.encoder,
                                        STN=self.stn,
                                        optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(checkpoint=self.ckpt,
                                                       directory='./checkpoints',
                                                       max_to_keep=5)
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print(f'Latest checkpoint restored at step {self.ckpt.step.numpy()}')
        

    def create_dataset(self, dir, batch_size):
        dataset = tf.data.Dataset.list_files(dir + '/*.jpg')
        dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(vgg_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size=256)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset


    def train(self):
        start = time.perf_counter()

        for epoch in range(0, self.num_epochs):
            epoch_start = time.perf_counter()

            for content_images, style_images in tf.data.Dataset.zip((self.content_dataset, self.style_dataset)):
                
                metrics = self.train_step(content_images, style_images)

                self.ckpt.step.assign_add(1)
                if self.ckpt.step % 3000 == 0:
 
                    if self.debug: 
                        """ Validating """
                        _, _, validate_output = self.stn([self.validate_content, self.validate_style, tf.constant([1.0])])
                        imsave(validate_output, f'./results/validate{self.ckpt.step.numpy()}.jpg')

                        self.write_summaries(metrics, self.ckpt.step.numpy())
                        self.update_metrics(metrics, self.ckpt.step.numpy())

                    self.ckpt_manager.save(checkpoint_number=self.ckpt.step.numpy())
                    print('=====================================')
                    print(f'     Step {self.ckpt.step.numpy()}: ckpt saved!    ')
                    print('=====================================\n')

            tf.saved_model.save(self.stn, os.path.join(self.model_path, 'model'))
            print('=====================================')
            print(f'       Epoch {epoch+1} saved!       ')
            print('=====================================\n')
            print(f'Time taken for epoch {epoch+1} is {time.perf_counter()-epoch_start:.2f} sec\n')
        print(f'Total Time taken is {time.perf_counter()-start:.2f} sec\n')


    @tf.function
    def train_step(self, content_images, style_images):
        with tf.GradientTape() as tape:
            style_features, latent_code, outputs = self.stn([content_images, style_images, tf.constant([1.0])])
            outputs = vgg_preprocess(outputs)
            outputs_features = self.encoder(outputs)

            c_loss = self.content_loss(outputs_features[-1], latent_code)
            s_loss = tf.reduce_sum([self.style_loss(output, style) for output, style in zip(outputs_features, style_features)])

            total_loss = self.content_weight * c_loss + self.style_weight * s_loss

        gradients = tape.gradient(total_loss, self.stn.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.stn.trainable_variables))

        metrics = (c_loss, s_loss, total_loss)
        return metrics


    def content_loss(self, output_features, content_target):
        return self.mse(output_features, content_target)


    def style_loss(self, output_features, style_target):
        output_mean, output_var = tf.nn.moments(output_features, axes=[1, 2], keepdims=True)
        style_mean, style_var = tf.nn.moments(style_target, axes=[1, 2], keepdims=True)
        output_std = tf.sqrt(output_var + 1e-6)
        style_std = tf.sqrt(style_var + 1e-6)

        return self.mse(output_mean, style_mean) + self.mse(output_std, style_std)


    def create_metrics(self):
        self.content_metric = Mean()
        self.style_metric = Mean()
        self.total_metric = Mean()


    def update_metrics(self, metrics, step):
        content_loss, style_loss, total_loss = metrics

        self.content_metric(content_loss)
        self.style_metric(style_loss)
        self.total_metric(total_loss)
        
        print(f'content_loss = {self.content_metric.result():.2f}')
        print(f'style_loss = {self.style_metric.result():.2f}')
        print(f'total_loss = {self.total_metric.result():.2f}')

        self.content_metric.reset_states()
        self.style_metric.reset_states()
        self.total_metric.reset_states()


    def create_summary_writer(self):
        import datetime
        self.summary_writer = tf.summary.create_file_writer(
            'log/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))


    def write_summaries(self, metrics, step):
        content_loss, style_loss, total_loss = metrics

        with self.summary_writer.as_default():
            tf.summary.scalar('content_loss', content_loss, step=step)
            tf.summary.scalar('style_loss', style_loss, step=step)
            # tf.summary.scalar('PSNR', psnr, step=step)