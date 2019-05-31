import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.framework import ops
import operator

class BaseStrategy(object):
    def __init__(self, cpu_device, gpu_devices, num_gpus):
        self._cpu_device = cpu_device
        self._gpu_devices = gpu_devices
        self._num_gpus = num_gpus

class LocalPSStrategy(BaseStrategy):
    def __init__(self, cpu_device, gpu_devices, num_gpus, use_staging=False):
        super().__init__(cpu_device, gpu_devices, num_gpus)
        self._use_staging = use_staging
        if use_staging:
            tf.logging.info('Using LocalPSStrategy - Staging')
            self._staging_put_ops = []
        else:
            tf.logging.info('Using LocalPSStrategy')

        self._param_server_device = gpu_devices[0]

        self._global_variable = {}
        self._local_variable = [
            dict() for _ in self._gpu_devices
        ]
        self._local_sizes = [0] * self._num_gpus

    def __call__(self, getter, name, *args, **kwargs):
        name_split = name.split('/', 2)
        device_index = int(name_split[1].split('_')[1])
        name_without_tower = name_split[0] + '/' + name_split[2]

        if (name_without_tower in self._global_variable):
            global_var = self._global_variable[name_without_tower]
        else:
            if (self._param_server_device == self._cpu_device):
                with tf.device(self._cpu_device):
                    global_var = getter(name_without_tower, *args, **kwargs)
            else:
                min_size_device, _ = min(enumerate(self._local_sizes), key=operator.itemgetter(1))
                with tf.device(self._gpu_devices[min_size_device]):
                    global_var = getter(name_without_tower, *args, **kwargs)
                self._local_sizes[min_size_device] += global_var.get_shape().num_elements()
            self._global_variable[name_without_tower] = global_var

        if self._use_staging:
            shape = kwargs['shape']
            dtype = kwargs['dtype']
            with tf.name_scope("Benchmark_Net/Input_Staging/Staging"):
                staging_var = data_flow_ops.StagingArea([dtype], [shape])
                put_op = staging_var.put(tf.identity(global_var))
                get_op = staging_var.get()[0]
                self._staging_put_ops.append(put_op)
                self._local_variable[device_index][name_without_tower] = get_op
        else:
            self._local_variable[device_index][name_without_tower] = global_var

        return self._local_variable[device_index][name_without_tower]

    def get_local_variable(self, index):
        return [v for k,v in self._local_variable[index].items()]
    
    def get_global_variable(self):
        return [v for k,v in self._global_variable.items()]
    
    def compute_gradient_and_apply(self, gradients_list, global_step, learning_rate, optimizer):

        if self._use_staging:
            input_staging_op = tf.group(self._staging_put_ops)
            gradients_put_op = []
            gradients_get_op = [
                list() for _ in self._gpu_devices
            ]
            for index, gradients in enumerate(gradients_list):
                with tf.device(self._gpu_devices[index]), tf.name_scope("Gradient_Staging/Staging"):
                    if isinstance(gradients[0], ops.IndexedSlices):
                        gradients[0] = tf.convert_to_tensor(gradients[0])
                    dtypes = [g.dtype for g in gradients]
                    shapes = [g.shape for g in gradients]
                    staging_var = data_flow_ops.StagingArea(dtypes, shapes)
                    gradients_put_op.append(staging_var.put(gradients))
                    gradients_get_op[index] = staging_var.get()
            gradients_list = gradients_get_op

        with tf.name_scope('Gradient_Update'):
            global_varis = self.get_global_variable()
            # print(global_varis[0])
            # for gg in gradients_list:
            #     print(gg[0])
            #     gg[0] = tf.convert_to_tensor(gg[0])
            #     print(gg[0])
            #global_varis = global_varis[1:]
            print(global_varis.__len__())
            if self._num_gpus > 1:
                apply_list = []
                for g_v in zip(*gradients_list, global_varis):
                    grads = g_v[:self._num_gpus]
                    varis = g_v[self._num_gpus]
                    #print(varis)
                    # Some variable in BatchNorm do not have gradient
                    if grads[0] != None:
                        with tf.device(varis.device):
                            if isinstance(grads[0], ops.IndexedSlices):
                                print(grads)
                            #     grads = tf.convert_to_tensor(grads)
                            average_grad = tf.multiply(tf.add_n(grads), 1.0 / self._num_gpus)
                            apply = optimizer.apply_gradients([(average_grad, varis)])
                            apply_list.append(apply)
                with tf.device(global_step.device):
                    apply_list.append(global_step.assign_add(1))
                apply_op = tf.group(apply_list)
            else:
                grads_and_varis = list(zip(gradients_list[0], global_varis))
                apply_op = optimizer.apply_gradients(grads_and_varis, global_step)

        if self._use_staging:
            return [input_staging_op, gradients_put_op, apply_op]
        else:
            return [apply_op]

# class LocalAllreduceStrategy(BaseStrategy):
#     def __init__(self, BenchMark):
#         super().__init__(BenchMark)
#         tf.logging.info('Using LocalAllreduceStrategy')

#         self._global_variable = {}
#         self._local_variable = [
#             dict() for _ in self._gpu_devices
#         ]

#         self._instance_key = 0

#     def __call__(self, getter, name, *args, **kwargs):
#         name_split = name.split('/', 2)
#         device_index = int(name_split[1].split('_')[1])
#         name_without_tower = name_split[0] + '/' + name_split[2]

#         if (name_without_tower in self._global_variable):
#             global_var = self._global_variable[name_without_tower]
#         else:
#             with tf.device(self._cpu_device):
#                 global_var = getter(name_without_tower, *args, **kwargs)
#             self._global_variable[name_without_tower] = global_var

#         local_var = getter(name, *args, **kwargs)

#         self._local_variable[device_index][name_without_tower] = local_var

#         return self._local_variable[device_index][name_without_tower]

#     def get_local_variable(self, index):
#         return [v for k,v in self._local_variable[index].items()]
    
#     def get_global_variable(self):
#         return [v for k,v in self._global_variable.items()]
    
#     def compute_gradient_and_apply(self, gradients_list, global_step, learning_rate):
#         optimizer = self.get_optimizer(learning_rate)

#         with tf.name_scope('Gradient_Update'):
#             if self._num_gpus > 1:
#                 apply_list = []

#                 local_variable_list = []
#                 for i in range(self._num_gpus):
#                     local_variable_list.append(self.get_local_variable(i))

#                 for g_v in zip(*gradients_list, *local_variable_list):
#                     instance_key = self._instance_key
#                     self._instance_key += 1
#                     grads = g_v[:self._num_gpus]
#                     varis = g_v[self._num_gpus:]
#                     for (grad, vari) in zip(grads, varis):
#                         with tf.device(vari.device):
#                             local_average_grad = collective_ops.all_reduce(grad, self._num_gpus, 1, instance_key, 'Add', 'Div')
#                             apply = optimizer.apply_gradients([(local_average_grad, vari)])
#                             apply_list.append(apply)
#                 with tf.device(global_step.device):
#                     apply_list.append(global_step.assign_add(1))
#                 apply_op = tf.group(apply_list)
#             else:
#                 global_varis = self.get_global_variable()
#                 grads_and_varis = list(zip(gradients_list[0], global_varis))
#                 apply_op = optimizer.apply_gradients(grads_and_varis, global_step)

#         return [apply_op]

class DistributedPSStrategy(BaseStrategy):
    def __init__(self, cpu_devices, gpu_devices, num_workers, num_gpus, use_staging=False):
        super().__init__(cpu_devices, gpu_devices, num_gpus)
        self._use_staging = use_staging
        if use_staging:
            tf.logging.info('Using DistributedPSStrategy - Staging')
            self._staging_put_ops = []
        else:
            tf.logging.info('Using DistributedPSStrategy')

        self._num_workers = num_workers
        self._total_gpus = self._num_workers * self._num_gpus

        self._param_server_device = cpu_devices

        self._global_variable = {}
        self._local_variable = [
            [dict() for _ in range(self._num_gpus)] for _ in range(self._num_workers)
        ]
        self._local_sizes = [0] * self._num_workers

    def __call__(self, getter, name, *args, **kwargs):
        name_split = name.split('/', 2)
        worker_index = int(name_split[1].split('_')[1])
        gpu_index = int(name_split[1].split('_')[2])
        name_without_tower = name_split[0] + '/' + name_split[2]

        if (name_without_tower in self._global_variable):
            global_var = self._global_variable[name_without_tower]
        else:
            min_size_device, _ = min(enumerate(self._local_sizes), key=operator.itemgetter(1))
            with tf.device(self._cpu_device[min_size_device]):
                global_var = getter(name_without_tower, *args, **kwargs)
            self._local_sizes[min_size_device] += global_var.get_shape().num_elements()
            self._global_variable[name_without_tower] = global_var

        if self._use_staging:
            shape = kwargs['shape']
            dtype = kwargs['dtype']
            with tf.name_scope("Benchmark_Net/Input_Staging/Staging"):
                staging_var = data_flow_ops.StagingArea([dtype], [shape])
                put_op = staging_var.put(tf.identity(global_var))
                get_op = staging_var.get()[0]
                self._staging_put_ops.append(put_op)
                self._local_variable[worker_index][gpu_index][name_without_tower] = get_op
        else:
            self._local_variable[worker_index][gpu_index][name_without_tower] = global_var

        return self._local_variable[worker_index][gpu_index][name_without_tower]

    def get_local_variable(self, worker_index, gpu_index):
        return [v for k,v in self._local_variable[worker_index][gpu_index].items()]
    
    def get_global_variable(self):
        return [v for k,v in self._global_variable.items()]
    
    def compute_gradient_and_apply(self, gradients_list, global_step, learning_rate, optimizer):

        if self._use_staging:
            input_staging_op = tf.group(self._staging_put_ops)
            gradients_put_op = []
            gradients_get_op = [
                list() for _ in range(self._total_gpus)
            ]
            for index, gradients in enumerate(gradients_list):
                worker_index = (int)(index / self._num_gpus)
                gpu_index = index % self._num_gpus
                with tf.device(self._gpu_devices[worker_index][gpu_index]), tf.name_scope("Gradient_Staging/Staging"):
                    dtypes = [g.dtype for g in gradients]
                    shapes = [g.shape for g in gradients]
                    staging_var = data_flow_ops.StagingArea(dtypes, shapes)
                    gradients_put_op.append(staging_var.put(gradients))
                    gradients_get_op[index] = staging_var.get()
            gradients_list = gradients_get_op

        with tf.name_scope('Gradient_Update'):
            global_varis = self.get_global_variable()

            apply_list = []
            for g_v in zip(*gradients_list, global_varis):
                grads = g_v[:self._total_gpus]
                varis = g_v[self._total_gpus]

                grad_sum_list = []
                for i in range(self._num_workers):
                    grads_in_worker = grads[i*self._num_gpus:(i+1)*self._num_gpus]
                    with tf.device(self._cpu_device[i]):
                        grad_sum_in_worker = tf.add_n(grads_in_worker)
                        grad_sum_list.append(grad_sum_in_worker)

                with tf.device(varis.device):
                    average_grad = tf.multiply(tf.add_n(grad_sum_list), 1.0 / self._total_gpus)
                    apply = optimizer.apply_gradients([(average_grad, varis)])
                    apply_list.append(apply)

            with tf.device(global_step.device):
                apply_list.append(global_step.assign_add(1))
            apply_op = tf.group(apply_list)

        if self._use_staging:
            return [input_staging_op, gradients_put_op, apply_op]
        else:
            return [apply_op]

# class DistributedAllreduceStrategy(BaseStrategy):
#     def __init__(self, BenchMark):
#         super().__init__(BenchMark)
#         tf.logging.info('Using DistributedAllreduceStrategy')

#         self._num_workers = BenchMark._num_workers
#         self._total_gpus = self._num_workers * self._num_gpus
#         self._param_server_device = BenchMark._param_server_device

#         self._global_variable = {}
#         self._local_variable = [
#             [dict() for _ in range(self._num_gpus)] for _ in range(self._num_workers)
#         ]

#         self._instance_key = 0

#     def __call__(self, getter, name, *args, **kwargs):
#         name_split = name.split('/', 2)
#         worker_index = int(name_split[1].split('_')[1])
#         gpu_index = int(name_split[1].split('_')[2])
#         name_without_tower = name_split[0] + '/' + name_split[2]

#         if (name_without_tower in self._global_variable):
#             global_var = self._global_variable[name_without_tower]
#         else:
#             with tf.device(self._cpu_device[0]):
#                 global_var = getter(name_without_tower, *args, **kwargs)
#             self._global_variable[name_without_tower] = global_var

#         local_var = getter(name, *args, **kwargs)

#         self._local_variable[worker_index][gpu_index][name_without_tower] = local_var

#         return self._local_variable[worker_index][gpu_index][name_without_tower]

#     def get_local_variable(self, worker_index, gpu_index):
#         return [v for k,v in self._local_variable[worker_index][gpu_index].items()]
    
#     def get_global_variable(self):
#         return [v for k,v in self._global_variable.items()]
    
#     def compute_gradient_and_apply(self, gradients_list, global_step, learning_rate):
#         optimizer = self.get_optimizer(learning_rate)

#         with tf.name_scope('Gradient_Update'):
#             apply_list = []

#             local_variable_list = []
#             for i in range(self._num_workers):
#                 for j in range(self._num_gpus):
#                     local_variable_list.append(self.get_local_variable(i, j))

#             for g_v in zip(*gradients_list, *local_variable_list):
#                 instance_key = self._instance_key
#                 self._instance_key += 1

#                 # grads = g_v[:self._total_gpus]
#                 # varis = g_v[self._total_gpus:]
#                 # for (grad, vari) in zip(grads, varis):
#                 #     with tf.device(vari.device):
#                 #         local_average_grad = collective_ops.all_reduce(grad, self._total_gpus, 1, instance_key, 'Add', 'Div')
#                 #         apply = optimizer.apply_gradients([(local_average_grad, vari)])
#                 #         apply_list.append(apply)

#                 for i in range(self._num_workers):
#                     grads = g_v[i*self._num_gpus:(i+1)*self._num_gpus]
#                     varis = g_v[self._total_gpus+i*self._num_gpus:self._total_gpus+(i+1)*self._num_gpus]
#                     with tf.device(self._cpu_device[i]):
#                         grads_sum = tf.add_n(grads)
#                         local_average_grad = collective_ops.all_reduce(grads_sum, self._num_workers, 1, instance_key, 'Add', 'Id')
#                     for vari in varis:
#                         with tf.device(vari.device):
#                             grad = tf.multiply(local_average_grad, 1.0 / self._num_gpus)
#                             apply = optimizer.apply_gradients([(grad, vari)])
#                             apply_list.append(apply)

#             with tf.device(global_step.device):
#                 apply_list.append(global_step.assign_add(1))
#             apply_op = tf.group(apply_list)

#         return [apply_op]
