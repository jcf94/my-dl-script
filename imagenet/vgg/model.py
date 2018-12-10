import tensorflow as tf

vgg_list = ['vgg11', 'vgg16', 'vgg19']

class Vgg(object):

    def __init__(self, image_size, data_format, batch_size, model):
        """ Init """

        if (model not in vgg_list):
            tf.errors.InvalidArgumentError(None, None, "Network Model not found.")

        self._conv_counter = 0
        self._pool_counter = 0
        self._affine_counter = 0
        self._image_size = image_size
        self._data_format = data_format
        self._model = model

        if self._data_format == 'NCHW':
            self._image_shape = [batch_size, 3, self._image_size, self._image_size]
        else:
            self._image_shape = [batch_size, self._image_size, self._image_size, 3]

    def _conv(self, inpOp, nIn, nOut, kH, kW, dH=1, dW=1, padType='SAME'):
        name = 'conv' + str(self._conv_counter)
        self._conv_counter += 1
        with tf.variable_scope(name):
            kernel = tf.get_variable('weights', [kH, kW, nIn, nOut], tf.float32, tf.truncated_normal_initializer(stddev=1e-1), trainable=True)
            biases = tf.get_variable('biases', [nOut], tf.float32, tf.zeros_initializer(), trainable=True)

            if self._data_format == 'NCHW':
                strides = [1, 1, dH, dW]
            else:
                strides = [1, dH, dW, 1]
            conv = tf.nn.conv2d(inpOp, kernel, strides, padding=padType,
                                data_format=self._data_format)
            
            bias = tf.reshape(tf.nn.bias_add(conv, biases,
                                            data_format=self._data_format),
                            conv.get_shape())
            conv1 = tf.nn.relu(bias, name='relu')
            return conv1

    def _affine(self, inpOp, nIn, nOut, need_relu=True):
        name = 'affine' + str(self._affine_counter)
        self._affine_counter += 1
        with tf.variable_scope(name):
            kernel = tf.get_variable('weights', [nIn, nOut], tf.float32, tf.truncated_normal_initializer(stddev=1e-1))
            biases = tf.get_variable('biases', [nOut], tf.float32, tf.zeros_initializer())
            xw_plus_b = tf.nn.xw_plus_b(inpOp, kernel, biases, name='xw_plus_b')
            if need_relu:
                affine1 = tf.nn.relu(xw_plus_b)
                return affine1
            else:
                return xw_plus_b

    def _mpool(self, inpOp, kH, kW, dH=2, dW=2):
        name = 'pool' + str(self._pool_counter)
        self._pool_counter += 1
        if self._data_format == 'NCHW':
            ksize = [1, 1, kH, kW]
            strides = [1, 1, dH, dW]
        else:
            ksize = [1, kH, kW, 1]
            strides = [1, dH, dW, 1]
        
        with tf.name_scope(name):
            pool = tf.nn.max_pool(inpOp,
                            ksize=ksize,
                            strides=strides,
                            padding='VALID',
                            data_format=self._data_format,
                            name='MaxPool')

        return pool

    def loss(self, logits, labels):
        with tf.name_scope('xentropy'):
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=logits,
                                                                labels=labels)
            loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        return loss

    def inference(self, images):
        with tf.variable_scope('Vgg', reuse=tf.AUTO_REUSE):
            if self._model == 'vgg11':
                return self._construct_vgg(images, [1, 1, 2, 2, 2])

            if self._model == 'vgg16':
                return self._construct_vgg(images, [2, 2, 3, 3, 3])

            if self._model == 'vgg19':
                return self._construct_vgg(images, [2, 2, 4, 4, 4])

    def _construct_vgg(self, images, num_conv_layers):
        """Build vgg architecture from blocks."""
        assert len(num_conv_layers) == 5
        last = images
        last_out = 3
        for _ in range(num_conv_layers[0]):
            last = self._conv(last, last_out, 64, 3, 3)
            last_out = 64
        last = self._mpool(last, 2, 2)
        for _ in range(num_conv_layers[1]):
            last = self._conv(last, last_out, 128, 3, 3)
            last_out = 128
        last = self._mpool(last, 2, 2)
        for _ in range(num_conv_layers[2]):
            last = self._conv(last, last_out, 256, 3, 3)
            last_out = 256
        last = self._mpool(last, 2, 2)
        for _ in range(num_conv_layers[3]):
            last = self._conv(last, last_out, 512, 3, 3)
            last_out = 512
        last = self._mpool(last, 2, 2)
        for _ in range(num_conv_layers[4]):
            last = self._conv(last, last_out, 512, 3, 3)
            last_out = 512
        last = self._mpool(last, 2, 2)
        last = tf.reshape(last, [-1, 512 * 7 * 7])
        last = self._affine(last, 512 * 7 * 7, 4096)
        last = self._affine(last, 4096, 4096)
        last = self._affine(last, 4096, 1000, False)

        return last
