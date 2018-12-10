import tensorflow.python.platform
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import data_flow_ops, state_ops, math_ops
from tensorflow.python.ops import variable_scope, control_flow_ops
from tensorflow.python.client import timeline
from datetime import datetime
import os
import math
import time
import six

from model_vgg import Vgg
from model_resnet import ResNet
from model_inception import Inception

# ----- CPU / GPU Set

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
CONFIG = tf.ConfigProto()
CONFIG.gpu_options.allow_growth=True
#CONFIG.log_device_placement=True

# -----

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('num_batches', 100,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('data_format', 'NCHW',
                           """The data format for Convnet operations.
                           Can be either NHWC or NCHW.
                           """)
tf.app.flags.DEFINE_string('local_parameter_device', 'gpu',
                            """""")
tf.app.flags.DEFINE_string('job_name', None,
                            """""")
tf.app.flags.DEFINE_integer('task_index', 0,
                            """""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """""")
tf.app.flags.DEFINE_string('model', "inception4",
                            """""")
tf.app.flags.DEFINE_string('trace_file', None,
                            """""")

class EstimatorBenchMark(object):
    def __init__(self, model, data_format, batch_size):
        """ init """

        if (model[:3] == 'vgg'):
            self._network = Vgg(data_format, model)
        elif (model[:6] == 'resnet'):
            self._network = ResNet(data_format, model)
        elif (model[:9] == 'inception'):
            self._network = Inception(data_format, model)

        self._model = model
        self._data_format = data_format
        self._batch_size = batch_size

        if FLAGS.job_name:
            self.worker_prefix = '/job:worker/task:%s' % FLAGS.task_index
        else:
            self.worker_prefix = ''
        
        self.cpu_device = '%s/cpu:0' % self.worker_prefix
        self.gpu_devices = [
            '%s/%s:%i' % (self.worker_prefix, 'gpu', i)
            for i in range(FLAGS.num_gpus)
        ]
        if FLAGS.local_parameter_device == 'gpu':
            self.param_server_device = self.gpu_devices[0]
        else:
            self.param_server_device = self.cpu_device

        self.replica_devices = [
            tf.train.replica_device_setter(
                worker_device=d,
                ps_device=self.param_server_device,
                ps_tasks=1) for d in self.gpu_devices
        ]

        self.global_step_device = self.param_server_device
        self.v_mgr = None

        def model_fn(features, labels, mode):

            last_layer = self._network.inference(features)

            predictions = {
                'classes': tf.argmax(input=last_layer, axis=1, name='classes'),
                'probabilities': tf.nn.softmax(last_layer, name='softmax_tensor')
            }

            if (mode == tf.estimator.ModeKeys.PREDICT):
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

            with tf.name_scope('xentropy'):
                cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=last_layer, labels=labels)
                loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

            with tf.name_scope('accuracy'):
                accuracy = tf.metrics.accuracy(labels=labels,
                predictions=predictions['classes'], name='accuracy')
                batch_accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions['classes']), tf.float32))
            
            eval_metric_ops = {'accuracy': accuracy}

            tf.summary.scalar('accuracy', batch_accuracy)

            if (mode == tf.estimator.ModeKeys.EVAL):
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

            logging_hook = tf.train.LoggingTensorHook({"loss": loss, "accuracy": batch_accuracy, "step": tf.train.get_global_step()}, every_n_iter=10)

            if (mode == tf.estimator.ModeKeys.TRAIN):
                with tf.name_scope('GD_Optimizer'):
                    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss, tf.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_step, eval_metric_ops=eval_metric_ops, training_hooks=[logging_hook])

        def fake_input_fn():

            if (self._model[:9] == "inception"):
                image_size = 299
            else:
                image_size = 224

            if self._data_format == 'NCHW':
                image_shape = [self._batch_size, 3, image_size, image_size]
            else:
                image_shape = [self._batch_size, image_size, image_size, 3]

            # ----------------------- Fake Input Images -----------------------
            with tf.device(self.cpu_device), tf.name_scope('Fake_Input_Images'):
                ori_images = tf.Variable(tf.random_normal(image_shape,
                                                        dtype=tf.float32,
                                                        stddev=1e-1), trainable=False)
                ori_labels = tf.Variable(tf.ones([self._batch_size], dtype=tf.int64), trainable=False)

            return ori_images, ori_labels

        self._model_fn = model_fn
        self._input_fn = fake_input_fn

    def run(self, steps):

        benchmark_estimator = tf.estimator.Estimator(self._model_fn, "estimator_train")
        benchmark_estimator.train(self._input_fn, steps=steps)

def run_benchmark():

    bench = EstimatorBenchMark(FLAGS.model, FLAGS.data_format, FLAGS.batch_size)

    tf.logging.set_verbosity(tf.logging.INFO)
    bench.run(FLAGS.num_batches)

def main(_):
    program_start_time = time.time()
    run_benchmark()
    program_end_time = time.time()
    print('Program finished, Total seconds: %s' % (program_end_time - program_start_time))

if __name__ == '__main__':
    tf.app.run()
