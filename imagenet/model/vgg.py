"""Vgg model configuration.

Includes multiple models: vgg11, vgg16, vgg19, corresponding to
  model A, D, and E in Table 1 of [1].

References:
[1]  Simonyan, Karen, Andrew Zisserman
     Very Deep Convolutional Networks for Large-Scale Image Recognition
     arXiv:1409.1556 (2014)
"""

import tensorflow as tf

from .convnet_builder import ConvNetBuilder

vgg_list = ['vgg11', 'vgg16', 'vgg19']

class Vgg(object):

    def __init__(self, data_format, model):
        """ Init """

        if (model not in vgg_list):
            tf.errors.InvalidArgumentError(None, None, "Network Model not found.")

        self._data_format = data_format
        self._model = model

    def inference(self, images):
        with tf.variable_scope(self._model, reuse=tf.AUTO_REUSE):
            if self._model == 'vgg11':
                return self._construct_vgg(images, [1, 1, 2, 2, 2])

            if self._model == 'vgg16':
                return self._construct_vgg(images, [2, 2, 3, 3, 3])

            if self._model == 'vgg19':
                return self._construct_vgg(images, [2, 2, 4, 4, 4])

    def _construct_vgg(self, images, num_conv_layers):
        """Build vgg architecture from blocks."""

        cnn = ConvNetBuilder(images, 3, True, True, self._data_format)

        assert len(num_conv_layers) == 5
        for _ in range(num_conv_layers[0]):
            cnn.conv(64, 3, 3)
        cnn.mpool(2, 2)
        for _ in range(num_conv_layers[1]):
            cnn.conv(128, 3, 3)
        cnn.mpool(2, 2)
        for _ in range(num_conv_layers[2]):
            cnn.conv(256, 3, 3)
        cnn.mpool(2, 2)
        for _ in range(num_conv_layers[3]):
            cnn.conv(512, 3, 3)
        cnn.mpool(2, 2)
        for _ in range(num_conv_layers[4]):
            cnn.conv(512, 3, 3)
        cnn.mpool(2, 2)
        cnn.reshape([-1, 512 * 7 * 7])
        cnn.affine(4096)
        cnn.dropout()
        cnn.affine(4096)
        cnn.dropout()
        last = cnn.affine(1001, activation=None)

        return last
