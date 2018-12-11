import tensorflow as tf
from datetime import datetime
import os
import time

from model.vgg import Vgg
from model.resnet import ResNet
from model.inception import Inception
from strategy import LocalPSStrategy

# ----- CPU / GPU Set

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
CONFIG = tf.ConfigProto()
CONFIG.gpu_options.allow_growth=True
#CONFIG.log_device_placement=True

# -----

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('num_batches', 200,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('data_format', 'NCHW',
                           """The data format for Convnet operations.
                           Can be either NHWC or NCHW.
                           """)
tf.app.flags.DEFINE_string('local_parameter_device', 'cpu',
                            """""")
tf.app.flags.DEFINE_string('job_name', None,
                            """""")
tf.app.flags.DEFINE_integer('task_index', 0,
                            """""")
tf.app.flags.DEFINE_integer('num_gpus', 2,
                            """""")
tf.app.flags.DEFINE_string('model', "vgg11",
                            """""")
tf.app.flags.DEFINE_string('trace_file', None,
                            """""")

class DatasetInitializerHook(tf.train.SessionRunHook):
    def __init__(self, iterator):
        self._iterator = iterator

    def begin(self):
        self._initializer = self._iterator._initializer

    def after_create_session(self, session, coord):
        del coord
        session.run(self._initializer)

class BenchMark(object):
    def __init__(self):
        """ init """

        # -------------- Model Config --------------

        self._model = FLAGS.model
        self._data_format = FLAGS.data_format

        if (self._model[:3] == 'vgg'):
            self._network = Vgg(self._data_format, self._model)
        elif (self._model[:6] == 'resnet'):
            self._network = ResNet(self._data_format, self._model)
        elif (self._model[:9] == 'inception'):
            self._network = Inception(self._data_format, self._model)

        self._batch_size = FLAGS.batch_size

        # -------------- Device Config --------------

        self._num_gpus = FLAGS.num_gpus

        if FLAGS.job_name:
            self.worker_prefix = '/job:worker/task:%s' % FLAGS.task_index
        else:
            self.worker_prefix = ''
        
        self.cpu_device = '%s/cpu:0' % self.worker_prefix
        self.gpu_devices = [
            '%s/%s:%i' % (self.worker_prefix, 'gpu', i)
            for i in range(self._num_gpus)
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

        self._global_step_device = self.param_server_device

        def model_fn(features, labels):

            last_layer = self._network.inference(features)

            with tf.name_scope('xentropy'):
                cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=last_layer, labels=labels)
                loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

            with tf.name_scope('accuracy'):
                classes = tf.argmax(input=last_layer, axis=1, name='classes')
                batch_accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, classes), tf.float32))

            return loss, batch_accuracy

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
                images = tf.data.Dataset.from_tensors(ori_images).repeat(100)
                labels = tf.data.Dataset.from_tensors(ori_labels).repeat(100)

                return tf.data.Dataset.zip((images, labels))

        self._model_fn = model_fn
        self._input_fn = fake_input_fn

    def build_network(self, hooks, chief_only_hooks, strategy):
        with tf.device(self._global_step_device):
            global_step = tf.train.get_or_create_global_step()
    
        with tf.device(self.cpu_device):
            input_data_iterator = self._input_fn().make_initializable_iterator('Input Data')

        # -------------- Network Model --------------
        gradients_list = []
        for index, gpu in enumerate(self.gpu_devices):
            with tf.device(gpu), tf.variable_scope('Tower_%i' % index, custom_getter=strategy):
                features, labels = input_data_iterator.get_next()
                loss, batch_accuracy = self._model_fn(features, labels)

                local_varis = strategy.get_local_variable(index)
                gradients = tf.gradients(loss, local_varis, aggregation_method=tf.AggregationMethod.DEFAULT)
                gradients_list.append(gradients)

        train_op = strategy.compute_gradient_and_apply(gradients_list, global_step)

        # -------------- Run Hooks --------------
        logging_hook = tf.train.LoggingTensorHook({"loss": loss, "accuracy": batch_accuracy,
                        "step": tf.train.get_global_step()}, every_n_iter=10)
        hooks.append(logging_hook)

        input_hook = DatasetInitializerHook(input_data_iterator)
        chief_only_hooks.append(input_hook)

        return train_op

    def run(self, steps):
        hooks = [tf.train.StopAtStepHook(steps)]
        chief_only_hooks = []

        strategy = LocalPSStrategy(self)

        with tf.variable_scope('Benchmark_Net'):
            train_op = self.build_network(hooks, chief_only_hooks, strategy)

        # -------------- Session Run --------------
        with tf.train.MonitoredTrainingSession(
            is_chief=True, checkpoint_dir='train', config=CONFIG,
            hooks=hooks, chief_only_hooks=chief_only_hooks) as sess:
            while not sess.should_stop():
                sess.run(train_op)

def run_benchmark():

    bench = BenchMark()

    tf.logging.set_verbosity(tf.logging.INFO)
    bench.run(FLAGS.num_batches)

def main(_):
    program_start_time = time.time()
    run_benchmark()
    program_end_time = time.time()
    print('Program finished, Total seconds: %s' % (program_end_time - program_start_time))

if __name__ == '__main__':
    tf.app.run()
