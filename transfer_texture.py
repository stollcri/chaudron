#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Based upon:
A Neural Algorithm of Artistic Style
Leon A. Gatys; Alexander S. Ecker; Matthias Bethge
https://arxiv.org/pdf/1508.06576.pdf

Variation in Python
Copyright © 2021 Christopher Stoll
"""

import argparse
import functools
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import time

from PIL import Image
from progress.bar import FillingCirclesBar
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras.preprocessing import image as kp_image

mpl.rcParams["figure.figsize"] = (10, 10)
mpl.rcParams["axes.grid"] = False


class Chaudron(object):
    def __init__(self, style_layer_weights):
        self.image_size = 512

        # Style layer we are interested in
        self.style_layers = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ]
        self.num_style_layers = len(self.style_layers)
        
        self.style_layer_weights = []
        for i, _l in enumerate(self.style_layers):
            if i < len(style_layer_weights):
                self.style_layer_weights.append(style_layer_weights[i])
            else:
                self.style_layer_weights.append(1.0)

        # Content layer where will pull our feature maps
        self.content_layers = ["block5_conv2"]
        self.num_content_layers = len(self.content_layers)

        # tf.compat.v1.disable_eager_execution()
        logging.info("Eager execution: {}".format(tf.executing_eagerly()))

    def load_image_file(self, image_file_name):
        image = Image.open(image_file_name)
        scale = self.image_size / max(image.size)
        image = image.resize(
            (round(image.size[0] * scale), round(image.size[1] * scale)),
            Image.ANTIALIAS,
        )
        image = kp_image.img_to_array(image)
        # broadcast the image array so it has a batch dimension
        image = np.expand_dims(image, axis=0)
        return image

    def load_image(self, image_file_name):
        image = self.load_image_file(image_file_name)
        image = tf.keras.applications.vgg19.preprocess_input(image)
        return image

    def postprocess_image(self, processed_image):
        image = processed_image.copy()

        if len(image.shape) == 4:
            image = np.squeeze(image, 0)

        if len(image.shape) != 3:
            logging.error(
                "Image must dimensionality must be [1, height, width, channel] or [height, width, channel]"
            )
            raise ValueError("Invalid input to deprocessing image")

        # perform the inverse of the preprocessing step
        image[:, :, 0] += 103.939
        image[:, :, 1] += 116.779
        image[:, :, 2] += 123.68
        image = image[:, :, ::-1]
        image = np.clip(image, 0, 255).astype("uint8")

        return image

    def save_image(self, image, file_name):
        img = Image.fromarray(image)
        img.save(file_name)

    def load_model(self):
        # Load VGG pretrained imagenet data
        vgg = tf.keras.applications.vgg19.VGG19(
            include_top=False, weights="imagenet", pooling="avg"
        )
        vgg.trainable = False
        # Get output layers corresponding to style and content layers
        style_outputs = [vgg.get_layer(name).output for name in self.style_layers]
        content_outputs = [vgg.get_layer(name).output for name in self.content_layers]
        model_outputs = style_outputs + content_outputs
        return models.Model(vgg.input, model_outputs)

    def get_content_loss(self, base_content, target):
        return tf.reduce_mean(tf.square(base_content - target))

    def gram_matrix(self, input_tensor):
        channels = int(input_tensor.shape[-1])
        a = tf.reshape(input_tensor, [-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(a, a, transpose_a=True)
        return gram / tf.cast(n, tf.float32)

    def get_style_loss(self, base_style, gram_target):
        gram_style = self.gram_matrix(base_style)
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def get_feature_representations(self, model, content_path, style_path):
        content_image = self.load_image(content_path)
        content_outputs = model(content_image)
        content_features = [
            content_layer[0]
            for content_layer in content_outputs[self.num_style_layers :]
        ]

        style_image = self.load_image(style_path)
        style_outputs = model(style_image)
        style_features = [
            style_layer[0] for style_layer in style_outputs[: self.num_style_layers]
        ]

        return style_features, content_features

    def compute_loss(
        self, model, loss_weights, init_image, gram_style_features, content_features
    ):
        """This function will compute the loss total loss.

        Arguments:
          model: The model that will give us access to the intermediate layers
          loss_weights: The weights of each contribution of each loss function.
            (style weight, content weight, and total variation weight)
          init_image: Our initial base image. This image is what we are updating with
            our optimization process. We apply the gradients wrt the loss we are
            calculating to this image.
          gram_style_features: Precomputed gram matrices corresponding to the
            defined style layers of interest.
          content_features: Precomputed outputs from defined content layers of
            interest.

        Returns:
          returns the total loss, style loss, content loss, and total variational loss
        """
        style_weight, content_weight = loss_weights

        # Feed our init image through our model. This will give us the content and
        # style representations at our desired layers. Since we're using eager
        # our model is callable just like any other function!
        model_outputs = model(init_image)

        style_output_features = model_outputs[: self.num_style_layers]
        content_output_features = model_outputs[self.num_style_layers :]

        style_score = 0
        content_score = 0

        # Accumulate style losses from all layers
        for i, (target_style, comb_style) in enumerate(
            zip(gram_style_features, style_output_features)
        ):
            style_score += self.style_layer_weights[i] * self.get_style_loss(
                comb_style[0], target_style
            )
            # print(f"style_score {style_score}")

        # Accumulate content losses from all layers
        for target_content, comb_content in zip(
            content_features, content_output_features
        ):
            content_score += self.get_content_loss(comb_content[0], target_content)
            # print(f"content_score {content_score}")

        style_score /= self.num_style_layers
        content_score /= self.num_content_layers
        # print("------")
        # print(f"style_score {style_score}")
        # print(f"content_score {content_score}")

        style_score *= style_weight
        content_score *= content_weight
        # print(f"style_score {style_score}")
        # print(f"content_score {content_score}")
        # print()

        # Get total loss
        loss = style_score + content_score
        return loss, style_score, content_score

    def compute_grads(self, cfg):
        with tf.GradientTape() as tape:
            all_loss = self.compute_loss(**cfg)
            # Compute gradients wrt input image
            total_loss = all_loss[0]
            return tape.gradient(total_loss, cfg["init_image"]), all_loss

    def run_style_transfer(
        self,
        content_path,
        style_path,
        folder,
        epochs,
        content_weight,
        style_weight,
        learning_rate,
        image_save_count,
    ):
        model = self.load_model()
        for layer in model.layers:
            layer.trainable = False

        # Get the style and content feature representations (from our specified intermediate layers)
        style_features, content_features = self.get_feature_representations(
            model, content_path, style_path
        )
        gram_style_features = [
            self.gram_matrix(style_feature) for style_feature in style_features
        ]

        init_image = self.load_image(content_path)
        init_image = tf.Variable(init_image, dtype=tf.float32)
        opt = tf.optimizers.Adam(learning_rate=0.05, beta_1=0.99, epsilon=1e-1)

        best_loss, best_img = float("inf"), None

        loss_weights = (style_weight, content_weight)
        cfg = {
            "model": model,
            "loss_weights": loss_weights,
            "init_image": init_image,
            "gram_style_features": gram_style_features,
            "content_features": content_features,
        }

        images_to_save = min(epochs, image_save_count)
        display_interval = epochs / images_to_save
        start_time = time.time()
        global_start = time.time()

        norm_means = np.array([103.939, 116.779, 123.68])
        min_vals = -norm_means
        max_vals = 255 - norm_means

        bar = FillingCirclesBar(
            f"Epoch {0}/{epochs} Loss: {0.0:4.2f} (content {0.0:4.2f}, style {0.0:4.2f})",
            max=epochs,
        )
        for epoch in range(epochs):
            grads, all_loss = self.compute_grads(cfg)
            loss, style_score, content_score = all_loss
            opt.apply_gradients([(grads, init_image)])
            clipped = tf.clip_by_value(init_image, min_vals, max_vals)
            init_image.assign(clipped)

            if loss < best_loss:
                best_loss = loss
                best_img = self.postprocess_image(init_image.numpy())

            if epoch % display_interval == 0:
                self.save_image(
                    best_img,
                    f"{folder}/e{epoch:05d}-l{best_loss:4.2f}-sl{style_score:4.2f}-cl{content_score:4.2f}.png",
                )

            bar.message = (
                f"Epoch {epoch}/{epochs}"
                f" Loss: {loss:4.2f}"
                f" (style {style_score:4.2f}"
                f" content {content_score:4.2f})"
                f" {(time.time() - start_time):.2f}s"
            )
            start_time = time.time()
            bar.next()
        bar.finish()
        logging.info("Total time: {:.4f}s".format(time.time() - global_start))

        return best_img, best_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate images using Deep Convolutional Generative Adversarial Network"
    )
    parser.add_argument(
        metavar="CONTENT_IMAGE",
        help="Content image (to be styled)",
        dest="content_image",
        default=None,
    )
    parser.add_argument(
        metavar="STYLE_IMAGE",
        help="Style image (to extract style from)",
        dest="style_image",
        default=None,
    )
    parser.add_argument(
        metavar="TARGET_DIR",
        help="Directory of resulting images",
        dest="target_dir",
        default=None,
    )
    parser.add_argument(
        "-v", "--verbose", help="verbose output", dest="verbose", action="store_true"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        help="The number of epochs to train (default=50000)",
        dest="epochs",
        default=5000,
    )
    parser.add_argument(
        "-c",
        "--content-weight",
        type=float,
        help="Weight to use for content (default=1000.0)",
        dest="content_weight",
        default=1000.0,
    )
    parser.add_argument(
        "-s",
        "--style-weight",
        type=float,
        help="Weight to use for style (default=0.01)",
        dest="style_weight",
        default=0.01,
    )
    parser.add_argument(
        "-l",
        "--learning-rate",
        type=float,
        help="Training learning rate (default=0.1)",
        dest="learning_rate",
        default=0.1,
    )
    parser.add_argument(
        "-i",
        "--image-save-count",
        type=int,
        help="The number of images to save (default=50)",
        dest="image_save_count",
        default=50,
    )
    parser.add_argument(
        "--style-layer-weights",
        type=float,
        nargs='+',
        help="Weights for style layers (default=[1.6, 0.8, 0.4, 0.2, 0.1])",
        dest="style_layer_weights",
        default=[1.6, 0.8, 0.4, 0.2, 0.1],
    )
    args = parser.parse_args()

    if (
        args.content_image is None
        or args.style_image is None
        or args.target_dir is None
    ):
        parser.print_usage()
        exit()

    if args.verbose:
        logging.basicConfig(format="%(message)s", level=logging.INFO)
    else:
        logging.basicConfig(format="%(message)s", level=logging.WARN)

    chaudron = Chaudron(args.style_layer_weights)
    chaudron.run_style_transfer(
        args.content_image,
        args.style_image,
        args.target_dir,
        args.epochs,
        args.content_weight,
        args.style_weight,
        args.learning_rate,
        args.image_save_count,
    )
