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

from vgg.vgg_model import Vgg

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
tf.app.flags.DEFINE_string('model', "vgg16",
                            """""")
tf.app.flags.DEFINE_boolean('staged_vars', False,
                            """""")
tf.app.flags.DEFINE_boolean('pipline_vars', False,
                            """""")
tf.app.flags.DEFINE_boolean('while_loop', False,
                            """ Use while_loop to run.""")
tf.app.flags.DEFINE_string('trace_file', None,
                            """""")
tf.app.flags.DEFINE_boolean('easgd', False,
                            """""")
tf.app.flags.DEFINE_integer('easgd_period', 10,
                            """""")

class VariableMgrEASGD(object):
    def __init__(self, bench):
        self.bench = bench
        self.gpu_devices = bench.gpu_devices
        self.vars_on_devices = [
            dict() for _ in self.gpu_devices
        ]
        self.center_vars_on_devices = {}
        self.center_vars_on_global = {}
        self.center_vars_on_global_by_name = dict()

    def create_outer_variable_scope(self, device_num):
        self.custom_getter = VariableEASGDGetter(self, device_num)
        return tf.variable_scope('tower_%i' % device_num, reuse=tf.AUTO_REUSE, custom_getter=self.custom_getter)
    
    def trainable_variables_on_device(self, device_num):
        vars_dict = self.vars_on_devices[device_num]
        params = []
        for name in vars_dict:
            params.append(vars_dict[name])
        return params

class VariableEASGDGetter(object):
    def __init__(self, variable_mgr, device_num):
        self.variable_mgr = variable_mgr
        self.device_num = device_num
    
    def __call__(self, getter, name, trainable, collections, *args, **kwargs):
        vars_on_devices = self.variable_mgr.vars_on_devices[self.device_num]
        center_vars_on_devices = self.variable_mgr.center_vars_on_devices
        center_vars_on_global = self.variable_mgr.center_vars_on_global
        center_vars_on_global_by_name = self.variable_mgr.center_vars_on_global_by_name

        if name in vars_on_devices:
            local_var = vars_on_devices[name]
            return local_var

        if trainable:
            with tf.device(self.variable_mgr.gpu_devices[self.device_num]):
                local_var = getter(name, trainable=True,
                                    collections=[ops.GraphKeys.LOCAL_VARIABLES],
                                    *args, **kwargs)
                vars_on_devices[name] = local_var

                local_center_variable = getter(
                    name='local_center_variable/%s' % name,
                    trainable=False,
                    collections=[ops.GraphKeys.LOCAL_VARIABLES],
                    *args, **kwargs)
                center_vars_on_devices[local_var] = local_center_variable

            if name in center_vars_on_global_by_name:
                center_vars_on_global[local_var] = center_vars_on_global_by_name[name]
                return local_var
            else:
                with tf.device(self.variable_mgr.bench.param_server_device):
                    global_center_variable = getter(
                        name='global_center_variable/%s' % name.split('/', 1)[1],
                        trainable=False,
                        collections=[ops.GraphKeys.GLOBAL_VARIABLES],
                        *args, **kwargs)
                    center_vars_on_global[local_var] = global_center_variable
                    center_vars_on_global_by_name[name] = global_center_variable

                return local_var
        else:
            return getter(name, trainable=trainable, collections=collections, *args, **kwargs)

class VariableMgrLocalFetchFromPS(object):
    def __init__(self, bench):
        self.bench = bench
        self.gpu_devices = bench.gpu_devices
        self.vars_on_devices = [
            dict() for _ in self.gpu_devices
        ]

    def trainable_variables_on_device(self,
                                    rel_device_num,
                                    writable=False):
        return self.custom_getter.trainable_variables_on_device(
            rel_device_num, writable=writable)

    def create_outer_variable_scope(self, device_num):
        self.custom_getter = VariableGetter(self, device_num)
        return tf.variable_scope(
            'v', reuse=bool(device_num), custom_getter=self.custom_getter)

class VariableGetter(object):
    def __init__(self, variable_mgr, device_num):
        self.variable_mgr = variable_mgr
        self.device_num = device_num
    
    def __call__(self, getter, name, *args, **kwargs):
        vars_on_devices = self.variable_mgr.vars_on_devices[self.device_num]

        if name in vars_on_devices:
            real_var = vars_on_devices[name]
            get_op = tf.identity(real_var)
            return get_op

        with tf.device(self.variable_mgr.bench.param_server_device):
            real_var = getter(name, *args, **kwargs)

        trainable = kwargs['trainable']

        if trainable:
            vars_on_devices[name] = real_var
            return tf.identity(real_var)
        else:
            return real_var

    def trainable_variables_on_device(self, rel_device_num,
                                        writable):
        params_refs = tf.trainable_variables()
        if writable:
            return params_refs
        params = []
        for param in params_refs:
            var_name = param.name.split(':')[0]
            _, var_get_op = self.variable_mgr.staging_vars_on_devices[rel_device_num][
                var_name]
            params.append(var_get_op)
        return params

class VariableMgrLocalFetchFromStagedPS(object):
    def __init__(self, gpu_devices):
        self.gpu_devices = gpu_devices
        self.staging_vars_on_devices = [
            dict() for _ in gpu_devices
        ]
        self.staging_areas_on_devices = [
            list() for _ in gpu_devices
        ]
        self.staging_grads_on_devices = dict()

    def trainable_variables_on_device(self,
                                    rel_device_num,
                                    writable=False):
        return self.custom_getter.trainable_variables_on_device(
            rel_device_num, writable=writable)

    def create_outer_variable_scope(self, device_num):
        self.custom_getter = StagedVariableGetter(self, device_num)
        return tf.variable_scope(
            'v', reuse=bool(device_num), custom_getter=self.custom_getter)

class StagedVariableGetter(object):
    def __init__(self, variable_mgr, device_num):
        self.variable_mgr = variable_mgr
        self.device_num = device_num
    
    def __call__(self, getter, name, *args, **kwargs):
        staging_ops = self.variable_mgr.staging_vars_on_devices[self.device_num]
        staging_areas = self.variable_mgr.staging_areas_on_devices[self.device_num]
        if name in staging_ops:
            put_op, get_op = staging_ops[name]
            return get_op
        real_var = getter(name, *args, **kwargs)
        trainable = kwargs['trainable']

        if trainable:
            shape = kwargs['shape']
            dtype = kwargs['dtype']
            var_to_stage = tf.identity(real_var)
            with tf.device(self.variable_mgr.gpu_devices[self.device_num]):
                staging_area = data_flow_ops.StagingArea([dtype], shapes=[shape])
                staging_areas.append([real_var, staging_area])
                put_op = staging_area.put([var_to_stage])
                get_op = staging_area.get()[0]
                staging_ops[name] = (put_op, get_op)
            return get_op
        else:
            return real_var

    def trainable_variables_on_device(self, rel_device_num,
                                        writable):
        params_refs = tf.trainable_variables()
        if writable:
            return params_refs
        params = []
        for param in params_refs:
            var_name = param.name.split(':')[0]
            _, var_get_op = self.variable_mgr.staging_vars_on_devices[rel_device_num][
                var_name]
            params.append(var_get_op)
        return params

class BenchMark(object):
    def __init__(self):
        """Init"""
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

    def build_network_easgd(self, image_size, TTT):

        print('|------ Build EASGD Network')

        grads_list = []
        if not self.v_mgr:
            self.v_mgr = VariableMgrEASGD(self)
        v_mgr = self.v_mgr

        #optimizers = []
        easgd_prepare = []
        conditional_updates = []

        def update_global_var():
            local_var = v_mgr.vars_on_devices[device_index]
            global_center_var = [v_mgr.center_vars_on_global[local_var[var]] for var in local_var]
            local_center_var = [v_mgr.center_vars_on_devices[local_var[var]] for var in local_var]

            var_update = []
            for lvar, gvar in zip(local_center_var, global_center_var):
                var_update.append(state_ops.assign(gvar, lvar))
            
            return control_flow_ops.group(*(var_update))

        for device_index, device_name in enumerate(self.replica_devices):

            network = Vgg(image_size, FLAGS.data_format, FLAGS.batch_size, FLAGS.model)
            with tf.name_scope('tower_%i' % device_index):
                # -----------------------------------------------------------

                with tf.device(self.gpu_devices[device_index]), tf.variable_scope('tower_%i/Gpu_%i_Own' % (device_index, device_index), reuse=tf.AUTO_REUSE):
                    local_step = tf.get_variable('local_step',
                                    initializer=tf.zeros([], tf.int64),
                                    trainable=False,
                                    collections=[ops.GraphKeys.LOCAL_VARIABLES])

                    images = tf.get_variable('gpu_cache_images', network._image_shape, tf.float32,
                                            tf.truncated_normal_initializer(1e-1), trainable=False)

                    #labels = tf.Variable(tf.ones([FLAGS.batch_size], dtype=tf.int64), trainable=False, name='gpu_cache_labels')
                    labels = tf.random_uniform(
                        [FLAGS.batch_size],
                        minval=0,
                        maxval=1000,
                        dtype=tf.int32,
                        name='synthetic_labels')

                # -----------------------------------------------------------
                with tf.device(device_name), v_mgr.create_outer_variable_scope(device_index):
                    last_layer = network.inference(images)
                    loss = network.loss(last_layer, labels)
                    varis = v_mgr.trainable_variables_on_device(device_index)
                    grads = tf.gradients(loss, varis, aggregation_method=tf.AggregationMethod.DEFAULT)

                    grads_and_varis = list(zip(grads, varis))
                    optimizer = tf.train.GradientDescentOptimizer(0.01).apply_gradients(grads_and_varis, local_step)
                    #optimizers.append(optimizer)

                    with ops.control_dependencies([optimizer]):
                        condition = math_ops.equal(math_ops.mod(tf.cast(local_step, tf.int32), TTT), 0)
                        conditional_update = control_flow_ops.cond(
                            condition, update_global_var, control_flow_ops.no_op)
                        conditional_updates.append(conditional_update)

            # -----------------------------------------------------------

            local_var = v_mgr.vars_on_devices[device_index]
            global_center_var = [v_mgr.center_vars_on_global[local_var[var]] for var in local_var]
            local_center_var = [v_mgr.center_vars_on_devices[local_var[var]] for var in local_var]

            for var, lvar, gvar in zip(local_var, local_center_var, global_center_var):
                easgd_prepare.append(state_ops.assign(local_var[var], gvar))
                easgd_prepare.append(state_ops.assign(lvar, gvar))

        return conditional_updates, easgd_prepare

    def build_network_hogwild(self, image_size):
        grads_list = []
        if not self.v_mgr:
            self.v_mgr = VariableMgrLocalFetchFromPS(self)
        v_mgr = self.v_mgr

        optimizers = []

        for device_index, device_name in enumerate(self.replica_devices):
            print(device_index, device_name)
            network = Vgg(image_size, FLAGS.data_format, FLAGS.batch_size, FLAGS.model)
            with tf.name_scope('tower_%i' % device_index):
                # -------------------------------------------------------------------

                with tf.device(self.gpu_devices[device_index]), tf.variable_scope('Gpu_%i_Own' % device_index, reuse=tf.AUTO_REUSE):
                    images = tf.get_variable('gpu_cache_images', network._image_shape, tf.float32,
                                            tf.truncated_normal_initializer(1e-1), trainable=False)

                    #labels = tf.Variable(tf.ones([FLAGS.batch_size], dtype=tf.int64), trainable=False, name='gpu_cache_labels')
                    labels = tf.random_uniform(
                        [FLAGS.batch_size],
                        minval=0,
                        maxval=1000,
                        dtype=tf.int32,
                        name='synthetic_labels')

                # -------------------------------------------------------------------
                with tf.device(device_name), v_mgr.create_outer_variable_scope(device_index):
                    last_layer = network.inference(images)
                    loss = network.loss(last_layer, labels)
                    varis = tf.trainable_variables()
                    grads = tf.gradients(loss, varis, aggregation_method=tf.AggregationMethod.DEFAULT)
                    # grads_list.append(grads)
                    grads_and_varis = list(zip(grads, varis))
                    optimizer = tf.train.GradientDescentOptimizer(0.01).apply_gradients(grads_and_varis, tf.train.get_or_create_global_step())
                    optimizers.append(optimizer)

        # -------------------------------------------------------------------

        # with tf.device(self.param_server_device):
        #     if FLAGS.num_gpus > 1:
        #         average_grads = []
        #         for grads in zip(*grads_list):
        #             average_grads.append(tf.multiply(tf.add_n(grads), 1.0 / FLAGS.num_gpus))
        #         grads_and_varis = list(zip(average_grads, varis))
        #     else:
        #         grads_and_varis = list(zip(grads_list[0], varis))
        #     optimizer = tf.train.GradientDescentOptimizer(0.01).apply_gradients(grads_and_varis, tf.train.get_or_create_global_step())

        return optimizers

    def build_network(self, image_size):
        grads_list = []
        if not self.v_mgr:
            self.v_mgr = VariableMgrLocalFetchFromPS(self)
        v_mgr = self.v_mgr

        for device_index, device_name in enumerate(self.replica_devices):
            network = Vgg(image_size, FLAGS.data_format, FLAGS.batch_size, FLAGS.model)
            with tf.name_scope('tower_%i' % device_index):
                # -------------------------------------------------------------------

                with tf.device(self.gpu_devices[device_index]), tf.variable_scope('Gpu_%i_Own' % device_index, reuse=tf.AUTO_REUSE):
                    images = tf.get_variable('gpu_cache_images', network._image_shape, tf.float32,
                                            tf.truncated_normal_initializer(1e-1), trainable=False)

                    #labels = tf.Variable(tf.ones([FLAGS.batch_size], dtype=tf.int64), trainable=False, name='gpu_cache_labels')
                    labels = tf.random_uniform(
                        [FLAGS.batch_size],
                        minval=0,
                        maxval=1000,
                        dtype=tf.int32,
                        name='synthetic_labels')

                # -------------------------------------------------------------------
                with tf.device(device_name), v_mgr.create_outer_variable_scope(device_index):
                    last_layer = network.inference(images)
                    loss = network.loss(last_layer, labels)
                    varis = tf.trainable_variables()
                    grads = tf.gradients(loss, varis, aggregation_method=tf.AggregationMethod.DEFAULT)

                    grads_list.append(grads)

        # -------------------------------------------------------------------

        with tf.device(self.param_server_device):
            if FLAGS.num_gpus > 1:
                average_grads = []
                for grads in zip(*grads_list):
                    average_grads.append(tf.multiply(tf.add_n(grads), 1.0 / FLAGS.num_gpus))
                grads_and_varis = list(zip(average_grads, varis))
            else:
                grads_and_varis = list(zip(grads_list[0], varis))
            optimizer = tf.train.GradientDescentOptimizer(0.01).apply_gradients(grads_and_varis, tf.train.get_or_create_global_step())

        return optimizer

    def build_staged_network(self, image_size):
        enqueue_ops = []
        gpu_compute_stage_ops = []
        grads_list = []
        grads_stage_list = []
        if not self.v_mgr:
            self.v_mgr = VariableMgrLocalFetchFromStagedPS(self.gpu_devices)
        v_mgr = self.v_mgr

        for device_index, device_name in enumerate(self.replica_devices):
            network = Vgg(image_size, FLAGS.data_format, FLAGS.batch_size, FLAGS.model)
            with tf.name_scope('tower_%i' % device_index):
                # -------------------------------------------------------------------

                with tf.device(self.gpu_devices[device_index]), tf.variable_scope('Gpu_%i_Own' % device_index, reuse=tf.AUTO_REUSE):
                    images = tf.get_variable('gpu_cache_images', network._image_shape, tf.float32,
                                            tf.truncated_normal_initializer(1e-1), trainable=False)

                    #labels = tf.Variable(tf.ones([FLAGS.batch_size], dtype=tf.int64), trainable=False, name='gpu_cache_labels')
                    labels = tf.random_uniform(
                        [FLAGS.batch_size],
                        minval=0,
                        maxval=1000,
                        dtype=tf.int32,
                        name='synthetic_labels')

                # -------------------------------------------------------------------
                with tf.device(device_name), v_mgr.create_outer_variable_scope(device_index):
                    last_layer = network.inference(images)
                    loss = network.loss(last_layer, labels)
                    # varis = tf.trainable_variables()
                    varis = v_mgr.trainable_variables_on_device(device_index)
                    grads = tf.gradients(loss, varis, aggregation_method=tf.AggregationMethod.DEFAULT)

                    #grads_list.append(grads)
                    grad_dtypes = [grad.dtype for grad in grads]
                    grad_shapes = [grad.shape for grad in grads]
                    
                    staging_grads_on_devices = v_mgr.staging_grads_on_devices
                    if device_name in staging_grads_on_devices:
                        grad_stage = staging_grads_on_devices[device_name]
                    else:
                        grad_stage = data_flow_ops.StagingArea(grad_dtypes, grad_shapes)
                        staging_grads_on_devices[device_name] = (grad_stage)
                    
                    grad_stage_op = grad_stage.put(grads)
                    grads_stage_list.append(grad_stage_op)
                    grads = grad_stage.get()

                    grads_list.append(grads)

        # -------------------------------------------------------------------

        varis = v_mgr.trainable_variables_on_device(device_index, writable=True)

        with tf.device(self.param_server_device):
            if FLAGS.num_gpus > 1:
                average_grads = []
                for grads in zip(*grads_list):
                    average_grads.append(tf.multiply(tf.add_n(grads), 1.0 / FLAGS.num_gpus))
                grads_and_varis = list(zip(average_grads, varis))
            else:
                grads_and_varis = list(zip(grads_list[0], varis))
            optimizer = tf.train.GradientDescentOptimizer(0.01).apply_gradients(grads_and_varis)

        for staging_ops in v_mgr.staging_vars_on_devices:
            gpu_compute_stage_ops.extend(
                [put_op for _, (put_op, _) in six.iteritems(staging_ops)])

        enqueue_ops.append(tf.group(*gpu_compute_stage_ops))
        enqueue_ops.append(tf.group(*(grads_stage_list)))

        with tf.device(self.global_step_device):
            gloabl_step = tf.train.get_or_create_global_step()
            with tf.control_dependencies([tf.group(optimizer, enqueue_ops)]):
                inc_global_step = gloabl_step.assign_add(1)

        return inc_global_step, enqueue_ops

    def do_step_run(self, image_size):
        print('|------ Start Per_step Run')

        if FLAGS.staged_vars:
            optimizer, enqueue_ops = self.build_staged_network(image_size)
        else:
            if FLAGS.easgd:
                optimizer, easgd_prepare = self.build_network_easgd(image_size, FLAGS.easgd_period)
            else:
                # optimizer = self.build_network(image_size)
                optimizer = self.build_network_hogwild(image_size)
            enqueue_ops = None

        init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]

        with tf.Session(config=CONFIG) as sess:

            tf.summary.FileWriter("./", sess.graph)

            num_steps_burn_in = 10
            total_duration = 0.0
            total_duration_squared = 0.0
            sess.run(init_op)

            if FLAGS.easgd:
                sess.run(easgd_prepare)

            if FLAGS.staged_vars:
                for i in range(len(enqueue_ops)):
                    sess.run(enqueue_ops[:(i + 1)])

            for i in range(FLAGS.num_batches + num_steps_burn_in):
                start_time = time.time()
                _ = sess.run(optimizer)
                duration = time.time() - start_time
                if i > num_steps_burn_in:
                    if not i % 10:
                        picps = FLAGS.num_gpus * FLAGS.batch_size / duration
                        print('%s: step %d, duration = %.3f, speed = %.3f pics/s' %
                            (datetime.now(), i - num_steps_burn_in, duration, picps))
                    total_duration += duration
                    total_duration_squared += duration * duration

            mn = total_duration / FLAGS.num_batches
            vr = total_duration_squared / FLAGS.num_batches - mn * mn
            sd = math.sqrt(vr)
            print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
                (datetime.now(), "Forward_backword", FLAGS.num_batches, mn, sd))
            picps = (FLAGS.num_gpus * FLAGS.num_batches * FLAGS.batch_size) / total_duration
            print('%.3f pics/s' % picps)

            if FLAGS.trace_file:
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _ = sess.run(optimizer, options=options, run_metadata=run_metadata)
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open(FLAGS.trace_file, 'w') as f:
                    f.write(chrome_trace)
                print('Chrome Trace File write in %s' % FLAGS.trace_file)

    def do_while_run(self, image_size):
        print('|------ Start while_loop Run')

        def cond(i):
            return i < FLAGS.num_batches

        if FLAGS.staged_vars:
            print('|------ Build Staged_vars network')
            fetches, enqueue_ops = self.build_staged_network(image_size)

            def body(i):
                fetches, _ = self.build_staged_network(image_size)

                put_ops_gropus = []
                for device_num in range(FLAGS.num_gpus):
                    with tf.device(self.gpu_devices[device_num]):
                        put_ops = []
                        for (real_var, staging_area) in self.v_mgr.staging_areas_on_devices[device_num]:
                            var_to_stage = tf.identity(real_var)
                            put_op = staging_area.put([var_to_stage])
                            put_ops.append(put_op)
                        g = tf.group(*put_ops)
                        put_ops_gropus.append(g)
                with tf.control_dependencies([fetches, *put_ops_gropus]):
                    return i+1

            with tf.device(self.param_server_device):
                loop = tf.while_loop(cond, body, [tf.constant(0)])
        else:
            print('|------ Build Normal network')
            fetches = self.build_network(image_size)

            def body(i):
                fetches = self.build_network(image_size)
                with tf.control_dependencies([fetches]):
                    return i+1

            with tf.device(self.param_server_device):
                loop = tf.while_loop(cond, body, [tf.constant(0)])

            enqueue_ops = None

        print('|------ Finish Model Building')

        init_op = tf.global_variables_initializer()

        with tf.Session(config=CONFIG) as sess:
            tf.summary.FileWriter("./", sess.graph)

            num_steps_burn_in = 10

            sess.run(init_op)
            if FLAGS.staged_vars:
                for i in range(len(enqueue_ops)):
                    sess.run(enqueue_ops[:(i + 1)])
            
            print('|------ Start Running Warmup')
            for i in range(num_steps_burn_in):
                start_time = time.time()
                sess.run(fetches)
                end_time = time.time()

                duration = end_time - start_time
                picps = (FLAGS.num_gpus * FLAGS.batch_size) / duration
                print('Cost Time: %s with %.3f pics/s' % (duration, picps))
            
            print('|------ Done Warmup')

            gloabl_step = tf.train.get_or_create_global_step()
            print('Global Step Start with: %s' % sess.run(gloabl_step))

            start_time = time.time()
            sess.run(loop)
            end_time = time.time()

            print('Global Step End with: %s' % sess.run(gloabl_step))
            total_duration = end_time - start_time
            print('Total Time: %s' % total_duration)
            picps = (FLAGS.num_gpus * FLAGS.num_batches * FLAGS.batch_size) / total_duration
            print('%.3f pics/s' % picps)

def run_benchmark():

    image_size = 224

    bench = BenchMark()

    # ----------------------- Fake Input Images -----------------------

    with tf.device(bench.cpu_device), tf.name_scope('Fake_Input_Images'):
        if FLAGS.data_format == 'NCHW':
            image_shape = [FLAGS.batch_size, 3, image_size, image_size]
        else:
            image_shape = [FLAGS.batch_size, image_size, image_size, 3]
        ori_images = tf.Variable(tf.random_normal(image_shape,
                                                dtype=tf.float32,
                                                stddev=1e-1), trainable=False)

        ori_labels = tf.Variable(tf.ones([FLAGS.batch_size], dtype=tf.int64), trainable=False)

    # -----------------------------------------------------------------

    if FLAGS.while_loop:
        return bench.do_while_run(image_size)
    else:
        return bench.do_step_run(image_size)

def main(_):
    program_start_time = time.time()
    run_benchmark()
    program_end_time = time.time()
    print('Program finished, Total seconds: %s' % (program_end_time - program_start_time))

if __name__ == '__main__':
    tf.app.run()
