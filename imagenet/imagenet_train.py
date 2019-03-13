import tensorflow as tf
from tensorflow.python.client import timeline

from datetime import datetime
import os
import time

from model.vgg import Vgg
from model.resnet import ResNet
from model.inception import Inception
from strategy import LocalPSStrategy, DistributedPSStrategy, LocalAllreduceStrategy, DistributedAllreduceStrategy
import process_imagenet_data.record_data_read as rdread

# PID = os.getpid()
# print('Program pid:', PID)
# print('Pause here to enter DBG')
# os.system('GREPDB="read"; /bin/bash -c "$GREPDB"')

# ----- CPU / GPU Set
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
CONFIG = tf.ConfigProto()
#CONFIG.gpu_options.allow_growth=True
#CONFIG.log_device_placement=True

# -----
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 64, """Batch size.""")
tf.app.flags.DEFINE_integer('num_epochs', 60, """""")
tf.app.flags.DEFINE_integer('num_batches', 100, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 2, """""")
tf.app.flags.DEFINE_string('model', 'resnet50', """""")
tf.app.flags.DEFINE_string('optimizer', 'sgd',
                            """ - sgd
                                - momentum
                            """)
tf.app.flags.DEFINE_float('learning_rate', 0.1, """""")
tf.app.flags.DEFINE_string('data_format', 'NCHW',
                           """The data format for Convnet operations.
                           Can be either NHWC or NCHW.
                           """)
tf.app.flags.DEFINE_string('local_parameter_device', 'cpu', """""")
tf.app.flags.DEFINE_string('trace_file', None, """""")
tf.app.flags.DEFINE_string('work_mode', 'test',
                            """ - test
                                - train
                                - validation
                            """)
tf.app.flags.DEFINE_string('strategy', 'ps',
                            """ - ps
                                - allreduce
                            """)
tf.app.flags.DEFINE_boolean('staged_vars', False, """""")

tf.app.flags.DEFINE_string('worker_hosts', None, 'Comma-separated list of target hosts')
tf.app.flags.DEFINE_integer('task_index', 0, """""")

class DatasetInitializerHook(tf.train.SessionRunHook):
    def __init__(self, iterator):
        self._iterator = iterator

    def begin(self):
        self._initializer = self._iterator.initializer

    def after_create_session(self, session, coord):
        del coord
        session.run(self._initializer)

class TraceHook(tf.train.SessionRunHook):
    """Hook to perform Traces every N steps."""

    def __init__(self, trace_file, target_step=50, trace_level=tf.RunOptions.FULL_TRACE):
        self._trace = target_step == 1
        self._trace_file = trace_file
        self._trace_level = trace_level
        self._target_step = target_step
        self._now_step = 1

    def begin(self):
        self._global_step_tensor = tf.train.get_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use _TraceHook.")

    def before_run(self, run_context):
        if self._trace:
            options = tf.RunOptions(trace_level=self._trace_level)
        else:
            options = None
        return tf.train.SessionRunArgs(fetches=[self._global_step_tensor], options=options)

    def after_run(self, run_context, run_values):
        if self._trace:
            self._trace = False
            fetched_timeline = timeline.Timeline(run_values.run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open(self._trace_file, 'w') as f:
                f.write(chrome_trace)
            print('Chrome Trace File write in %s' % FLAGS.trace_file)

        self._now_step += 1
        if self._now_step == self._target_step:
            self._trace = True

def create_done_queue(i):
    """Queue used to signal death for i'th ps shard. Intended to have 
    all workers enqueue an item onto it to signal doneness."""
    
    with tf.device("/job:worker/replica:0/task:%d" % (i)):
        return tf.FIFOQueue(1, tf.int32, shared_name="done_queue_"+
                            str(i))

def create_done_queues(n):
    return [create_done_queue(i) for i in range(1, n)]

class DistributedEndHook(tf.train.SessionRunHook):
    def __init__(self, num_workers):
        self._num_workers = num_workers
    
    def begin(self):
        self._done_queue = [q.enqueue(1) for q in create_done_queues(self._num_workers)]

    def end(self, session):
        i = 0
        for op in self._done_queue:
            session.run(op)
            i = i+1
            print('Worker %i Closed' % i)

DATA_DIR = "process_imagenet_data/record_data/"
class ImageNet_Data(object):
    def __init__(self, name, subset):
        assert subset in self.available_subsets()
        self.name = name
        self.subset = subset

    def available_subsets(self):
        return ['train', 'validation']

    def num_examples_per_epoch(self):
        if self.subset == 'train':
            return 1281167
        elif self.subset == 'validation':
            return 50000

    def dataset(self):
        tf_record_pattern = os.path.join(DATA_DIR, '%s-*' % self.subset)
        data_files = tf.gfile.Glob(tf_record_pattern)

        if not data_files:
            print("Error")
            exit(-1)

        dataset = tf.data.TFRecordDataset(data_files, buffer_size=10000, num_parallel_reads=4)
        return dataset
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

        if FLAGS.worker_hosts:
            self._worker_hosts = FLAGS.worker_hosts.split(",")
            self._num_workers = self._worker_hosts.__len__()
            self._worker_prefix = ['/job:worker/replica:0/task:%s' % i for i in range(self._num_workers)]

            self.cpu_device = ['%s/cpu:0' % prefix for prefix in self._worker_prefix]
            self.gpu_devices = [
                ['%s/%s:%i' % (prefix, 'device:GPU', i)
                for i in range(self._num_gpus)]
                for prefix in self._worker_prefix
            ]

            self._param_server_device = self.cpu_device
            self._global_step_device = self._param_server_device[0]

            if FLAGS.strategy == 'ps':
                self._strategy = DistributedPSStrategy(self, FLAGS.staged_vars)
            elif FLAGS.strategy == 'allreduce':
                self._strategy = DistributedAllreduceStrategy(self)
            else:
                tf.logging.error("Strategy not found.")
                return
        else:
            self._worker_prefix = None
            self.cpu_device = '/device:CPU:0'
            self.gpu_devices = [
                '/device:GPU:%i' % i
                for i in range(self._num_gpus)
            ]

            if FLAGS.local_parameter_device == 'gpu':
                self._param_server_device = self.gpu_devices[0]
            else:
                self._param_server_device = self.cpu_device
            self._global_step_device = self._param_server_device

            if FLAGS.strategy == 'ps':
                self._strategy = LocalPSStrategy(self, FLAGS.staged_vars)
            elif FLAGS.strategy == 'allreduce':
                self._strategy = LocalAllreduceStrategy(self)
            else:
                tf.logging.error("Strategy not found.")
                return

        # -------------- Model_fn & Input_fn --------------
        def model_fn(features, labels, input_data_iterator):

            last_layer = self._network.inference(features)

            # # ------------
            # temp_sess = tf.Session()
            # temp_sess.run(input_data_iterator.initializer)
            # temp_sess.run(tf.global_variables_initializer())
            # a1 = temp_sess.run([last_layer])
            # print(a1)
            # os.system('GREPDB="read"; /bin/bash -c "$GREPDB"')
            # # ------------

            with tf.name_scope('xentropy'):
                cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=last_layer, labels=labels)
                loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
            with tf.device(self.cpu_device):
                top_1_op = tf.reduce_sum(
                    tf.cast(tf.nn.in_top_k(last_layer, labels, 1), tf.float32))
                top_5_op = tf.reduce_sum(
                    tf.cast(tf.nn.in_top_k(last_layer, labels, 5), tf.float32))

            return loss, top_1_op, top_5_op

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
            if self._worker_prefix:
                image_device = self.cpu_device[0]
            else:
                image_device = self.cpu_device

            with tf.device(image_device), tf.name_scope('Fake_Input_Images'):
                ori_images = tf.Variable(tf.random_normal(image_shape,
                                                        dtype=tf.float32,
                                                        stddev=1e-1), trainable=False)
                ori_labels = tf.Variable(tf.ones([self._batch_size], dtype=tf.int64), trainable=False)
                images = tf.data.Dataset.from_tensors(ori_images).repeat()
                labels = tf.data.Dataset.from_tensors(ori_labels).repeat()

                return tf.data.Dataset.zip((images, labels)).prefetch(1)

        def imagenet_input_fn():
            if (self._model[:9] == "inception"):
                image_size = 299
            else:
                image_size = 224
            self._image_size = image_size

            if self._data_format == 'NCHW':
                image_shape = [self._batch_size, 3, image_size, image_size]
            else:
                image_shape = [self._batch_size, image_size, image_size, 3]

            if self._worker_prefix:
                image_device = self.cpu_device[0]
            else:
                image_device = self.cpu_device

            def map_fn(example_serialized):
                image_buffer, label_index, bbox, _ = rdread.parse_example_proto(example_serialized)
                return rdread.simple_process(image_buffer, bbox, image_size, image_size, 3, True), label_index

            with tf.device(image_device), tf.name_scope('ImageNet_Input_Images'):
                data = ImageNet_Data('ImageNet', FLAGS.work_mode)
                dataset = data.dataset()
                dataset = dataset.map(map_fn,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
                dataset = dataset.shuffle(buffer_size=10000)
                dataset = dataset.batch(batch_size=self._batch_size)
                dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
                dataset = dataset.repeat(FLAGS.num_epochs)
                return dataset

        self._model_fn = model_fn
        if FLAGS.work_mode == 'test':
            self._input_fn = fake_input_fn
        elif FLAGS.work_mode in ['train', 'validation']:
            self._input_fn = imagenet_input_fn
        else:
            print("work_mode Error")
            exit(-1)

    def build_network(self, hooks, chief_only_hooks, strategy):

        with tf.device(self._global_step_device):
            global_step = tf.train.get_or_create_global_step()

        gradients_list = []
        loss_list = []
        top_1_list = []
        top_5_list = []
        if self._worker_prefix:
            with tf.device(self.cpu_device[0]):
                input_data_iterator = self._input_fn().make_initializable_iterator('Input Data')
            # -------------- Network Model --------------
            for worker_index, gpu_in_worker in enumerate(self.gpu_devices):
                for gpu_index, gpu in enumerate(gpu_in_worker):
                    with tf.device(gpu), tf.variable_scope('Tower_%i_%i' % (worker_index, gpu_index), custom_getter=strategy):
                        features, labels = input_data_iterator.get_next()
                        loss, batch_accuracy = self._model_fn(features, labels)
                        local_varis = strategy.get_local_variable(worker_index, gpu_index)
                        gradients = tf.gradients(loss, local_varis, aggregation_method=tf.AggregationMethod.DEFAULT)
                        gradients_list.append(gradients)
        else:
            with tf.device(self.cpu_device):
                input_data_iterator = self._input_fn().make_initializable_iterator('Input Data')
            # -------------- Network Model --------------
            for index, gpu in enumerate(self.gpu_devices):
                with tf.device(gpu), tf.variable_scope('Tower_%i' % index, custom_getter=strategy):
                    features, labels = input_data_iterator.get_next()
                    if FLAGS.work_mode == 'train' and self._data_format == 'NCHW':
                        features = tf.reshape(features, [self._batch_size, self._image_size, self._image_size, 3])
                        labels = tf.reshape(labels, [self._batch_size])
                        features = tf.transpose(features, [0, 3, 1, 2])
                    loss, top_1, top_5 = self._model_fn(features, labels, input_data_iterator)
                    loss_list.append(loss)
                    top_1_list.append(top_1)
                    top_5_list.append(top_5)
                    local_varis = strategy.get_local_variable(index)
                    # print(local_varis)
                    gradients = tf.gradients(loss, local_varis, aggregation_method=tf.AggregationMethod.DEFAULT)
                    gradients_list.append(gradients)

        train_op = strategy.compute_gradient_and_apply(gradients_list, global_step, self.get_learning_rate(global_step))

        # -------------- Run Hooks --------------
        if FLAGS.work_mode == 'test':
            log_iter = 10
        else:
            log_iter = 10
        with tf.name_scope('average_loss'):
            average_loss = tf.reduce_mean(loss_list)
        with tf.name_scope('top_accuracy'):
            total_top_1 = tf.reduce_sum(top_1_list)
            total_top_5 = tf.reduce_sum(top_5_list)
        logging_hook = tf.train.LoggingTensorHook({"loss": average_loss,
                        "top_1": total_top_1, "top_5": total_top_5,
                        "step": tf.train.get_global_step()}, every_n_iter=log_iter)
        hooks.append(logging_hook)

        input_hook = DatasetInitializerHook(input_data_iterator)
        chief_only_hooks.append(input_hook)

        return train_op

    def run(self, steps):
        if self._worker_prefix:
            chief_only_hooks = [DistributedEndHook(self._num_workers)]
        else:
            chief_only_hooks = []

        if FLAGS.work_mode == 'test':
            hooks = [tf.train.StopAtStepHook(steps)]
        else:
            hooks = []
        if FLAGS.trace_file:
            hooks.append(TraceHook(FLAGS.trace_file))

        if self._worker_prefix:
            cluster = tf.train.ClusterSpec({"worker": self._worker_hosts})

            server = tf.train.Server(cluster,
                        job_name='worker',
                        task_index=FLAGS.task_index,
                        protocol="grpc+verbs")
            
            if FLAGS.task_index and FLAGS.strategy != "allreduce":
                with tf.Session(server.target) as sess:
                    sess.run(create_done_queue(FLAGS.task_index).dequeue())
                print('Worker %i Ready to Close' % FLAGS.task_index)
                return

            target = server.target
            is_chief = FLAGS.task_index == 0
        else:
            target = None
            is_chief = True

        with tf.variable_scope('Benchmark_Net'):
            train_op = self.build_network(hooks, chief_only_hooks, self._strategy)

        # -------------- Session Run --------------
        with tf.train.MonitoredTrainingSession(target,
            is_chief=is_chief, summary_dir='train', config=CONFIG,
            hooks=hooks, chief_only_hooks=chief_only_hooks) as sess:
            # -------------- Warmup & Pre Load Stage --------------
            if train_op.__len__() > 1:
                for i in range(len(train_op)):
                    sess.run_step_fn(lambda step_context: step_context.session.run(train_op[:i]))
                tf.logging.info('Staging Pre Load')

            tf.logging.info('Training Start')

            while not sess.should_stop():
                res = sess.run(train_op)

    def get_learning_rate(self, global_step):
        rescaled_lr = FLAGS.learning_rate
        num_batches_per_epoch = (1281167 / self._batch_size)
        boundaries = [int(num_batches_per_epoch * x) for x in [30, 60, 80, 90]]
        values = [1, 0.1, 0.01, 0.001, 0.0001]
        values = [rescaled_lr * v for v in values]
        lr = tf.train.piecewise_constant(global_step, boundaries, values)
        warmup_steps = int(num_batches_per_epoch * 5)
        warmup_lr = (
            rescaled_lr * tf.cast(global_step, tf.float32) / tf.cast(
                warmup_steps, tf.float32))
        return tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)

def run_benchmark():

    tf.logging.set_verbosity(tf.logging.INFO)

    bench = BenchMark()

    bench.run(FLAGS.num_batches)

def main(_):
    program_start_time = time.time()
    run_benchmark()
    program_end_time = time.time()
    print('Program finished, Total seconds: %s' % (program_end_time - program_start_time))

if __name__ == '__main__':
    tf.app.run()
