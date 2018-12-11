"""Resnet model configuration.

References:
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Deep Residual Learning for Image Recognition
  arXiv:1512.03385 (2015)

  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Identity Mappings in Deep Residual Networks
  arXiv:1603.05027 (2016)

  Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy,
  Alan L. Yuille
  DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,
  Atrous Convolution, and Fully Connected CRFs
  arXiv:1606.00915 (2016)
"""

import tensorflow as tf

from .convnet_builder import ConvNetBuilder

resnet_list = [ 'resnet50', 'resnet50_v1.5', 'resnet50_v2',
                'resnet101', 'resnet101_v2',
                'resnet152', 'resnet152_v2']

class ResNet(object):

    def __init__(self, data_format, model):
        """ Init """

        if (model not in resnet_list):
            tf.errors.InvalidArgumentError(None, None, "Network Model not found.")

        self._data_format = data_format
        self._model = model.split("_")[0]

        if 'v2' in model:
            self.version = 'v2'
        elif 'v1.5' in model:
            self.version = 'v1.5'
        else:
            self.version = 'v1'

    def inference(self, images):
        with tf.variable_scope(self._model, reuse=tf.AUTO_REUSE):
            if self._model == 'resnet50':
                return self._construct_resnet(images, [3, 4, 6, 3])

            if self._model == 'resnet101':
                return self._construct_resnet(images, [3, 4, 23, 3])

            if self._model == 'resnet152':
                return self._construct_resnet(images, [3, 8, 36, 3])

    def _construct_resnet(self, images, num_conv_layers):
        """Build vgg architecture from blocks."""

        cnn = ConvNetBuilder(images, 3, True, True, self._data_format)

        def bottleneck_block_v1(cnn, depth, depth_bottleneck, stride):
            """Bottleneck block with identity short-cut for ResNet v1.

            Args:
                cnn: the network to append bottleneck blocks.
                depth: the number of output filters for this bottleneck block.
                depth_bottleneck: the number of bottleneck filters for this block.
                stride: Stride used in the first layer of the bottleneck block.
            """
            input_layer = cnn.top_layer
            in_size = cnn.top_size
            name_key = 'resnet_v1'
            name = name_key + str(cnn.counts[name_key])
            cnn.counts[name_key] += 1

            with tf.variable_scope(name):
                if depth == in_size:
                    if stride == 1:
                        shortcut = input_layer
                    else:
                        shortcut = cnn.apool(
                            1, 1, stride, stride, input_layer=input_layer,
                            num_channels_in=in_size)
                else:
                    shortcut = cnn.conv(
                        depth, 1, 1, stride, stride, activation=None,
                        use_batch_norm=True, input_layer=input_layer,
                        num_channels_in=in_size, bias=None)
                cnn.conv(depth_bottleneck, 1, 1, stride, stride,
                        input_layer=input_layer, num_channels_in=in_size,
                        use_batch_norm=True, bias=None)
                cnn.conv(depth_bottleneck, 3, 3, 1, 1, mode='SAME_RESNET',
                        use_batch_norm=True, bias=None)
                res = cnn.conv(depth, 1, 1, 1, 1, activation=None,
                            use_batch_norm=True, bias=None)
                output = tf.nn.relu(shortcut + res)
                cnn.top_layer = output
                cnn.top_size = depth

        def bottleneck_block_v1_5(cnn, depth, depth_bottleneck, stride):
            """Bottleneck block with identity short-cut for ResNet v1.5.

            ResNet v1.5 is the informal name for ResNet v1 where stride 2 is used in the
            first 3x3 convolution of each block instead of the first 1x1 convolution.

            First seen at https://github.com/facebook/fb.resnet.torch. Used in the paper
            "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
            (arXiv:1706.02677v2) and by fast.ai to train to accuracy in 45 epochs using
            multiple image sizes.

            Args:
                cnn: the network to append bottleneck blocks.
                depth: the number of output filters for this bottleneck block.
                depth_bottleneck: the number of bottleneck filters for this block.
                stride: Stride used in the first layer of the bottleneck block.
            """
            input_layer = cnn.top_layer
            in_size = cnn.top_size
            name_key = 'resnet_v1.5'
            name = name_key + str(cnn.counts[name_key])
            cnn.counts[name_key] += 1

            with tf.variable_scope(name):
                if depth == in_size:
                    if stride == 1:
                        shortcut = input_layer
                    else:
                        shortcut = cnn.apool(
                            1, 1, stride, stride, input_layer=input_layer,
                            num_channels_in=in_size)
                else:
                    shortcut = cnn.conv(
                        depth, 1, 1, stride, stride, activation=None,
                        use_batch_norm=True, input_layer=input_layer,
                        num_channels_in=in_size, bias=None)
                cnn.conv(depth_bottleneck, 1, 1, 1, 1,
                        input_layer=input_layer, num_channels_in=in_size,
                        use_batch_norm=True, bias=None)
                cnn.conv(depth_bottleneck, 3, 3, stride, stride, mode='SAME_RESNET',
                        use_batch_norm=True, bias=None)
                res = cnn.conv(depth, 1, 1, 1, 1, activation=None,
                            use_batch_norm=True, bias=None)
                output = tf.nn.relu(shortcut + res)
                cnn.top_layer = output
                cnn.top_size = depth

        def bottleneck_block_v2(cnn, depth, depth_bottleneck, stride):
            """Bottleneck block with identity short-cut for ResNet v2.

            The main difference from v1 is that a batch norm and relu are done at the
            start of the block, instead of the end. This initial batch norm and relu is
            collectively called a pre-activation.

            Args:
                cnn: the network to append bottleneck blocks.
                depth: the number of output filters for this bottleneck block.
                depth_bottleneck: the number of bottleneck filters for this block.
                stride: Stride used in the first layer of the bottleneck block.
            """
            input_layer = cnn.top_layer
            in_size = cnn.top_size
            name_key = 'resnet_v2'
            name = name_key + str(cnn.counts[name_key])
            cnn.counts[name_key] += 1

            preact = cnn.batch_norm()
            preact = tf.nn.relu(preact)
            with tf.variable_scope(name):
                if depth == in_size:
                    if stride == 1:
                        shortcut = input_layer
                    else:
                        shortcut = cnn.apool(
                            1, 1, stride, stride, input_layer=input_layer,
                            num_channels_in=in_size)
                else:
                    shortcut = cnn.conv(
                        depth, 1, 1, stride, stride, activation=None, use_batch_norm=False,
                        input_layer=preact, num_channels_in=in_size, bias=None)
                cnn.conv(depth_bottleneck, 1, 1, stride, stride,
                        input_layer=preact, num_channels_in=in_size,
                        use_batch_norm=True, bias=None)
                cnn.conv(depth_bottleneck, 3, 3, 1, 1, mode='SAME_RESNET',
                        use_batch_norm=True, bias=None)
                res = cnn.conv(depth, 1, 1, 1, 1, activation=None,
                            use_batch_norm=False, bias=None)
                output = shortcut + res
                cnn.top_layer = output
                cnn.top_size = depth

        def bottleneck_block(cnn, depth, depth_bottleneck, stride, version):
            """Bottleneck block with identity short-cut.

            Args:
                cnn: the network to append bottleneck blocks.
                depth: the number of output filters for this bottleneck block.
                depth_bottleneck: the number of bottleneck filters for this block.
                stride: Stride used in the first layer of the bottleneck block.
                version: version of ResNet to build.
            """
            if version == 'v2':
                bottleneck_block_v2(cnn, depth, depth_bottleneck, stride)
            elif version == 'v1.5':
                bottleneck_block_v1_5(cnn, depth, depth_bottleneck, stride)
            else:
                bottleneck_block_v1(cnn, depth, depth_bottleneck, stride)

        assert len(num_conv_layers) == 4
        cnn.use_batch_norm = True
        cnn.batch_norm_config = {'decay': 0.9, 'epsilon': 1e-5, 'scale': True}
        cnn.conv(64, 7, 7, 2, 2, mode='SAME_RESNET', use_batch_norm=True)
        cnn.mpool(3, 3, 2, 2, mode='SAME')
        for _ in range(num_conv_layers[0]):
            bottleneck_block(cnn, 256, 64, 1, self.version)
        for i in range(num_conv_layers[1]):
            stride = 2 if i == 0 else 1
            bottleneck_block(cnn, 512, 128, stride, self.version)
        for i in range(num_conv_layers[2]):
            stride = 2 if i == 0 else 1
            bottleneck_block(cnn, 1024, 256, stride, self.version)
        for i in range(num_conv_layers[3]):
            stride = 2 if i == 0 else 1
            bottleneck_block(cnn, 2048, 512, stride, self.version)
        if self.version == 'v2':
            cnn.batch_norm()
            cnn.top_layer = tf.nn.relu(cnn.top_layer)
        cnn.spatial_mean()
        last = cnn.affine(1000, activation=None)

        return last
