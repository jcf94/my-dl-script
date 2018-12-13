import tensorflow as tf
import operator

class LocalPSStrategy(object):
    def __init__(self, BenchMark):
        self._cpu_device = BenchMark.cpu_device
        self._num_gpus = BenchMark._num_gpus
        self._gpu_devices = BenchMark.gpu_devices
        self._param_server_device = BenchMark._param_server_device

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

        self._local_variable[device_index][name_without_tower] = global_var

        return self._local_variable[device_index][name_without_tower]

    def get_local_variable(self, index):
        return [v for k,v in self._local_variable[index].items()]
    
    def get_global_variable(self):
        return [v for k,v in self._global_variable.items()]
    
    def compute_gradient_and_apply(self, gradients_list, global_step):
        optimizer = tf.train.GradientDescentOptimizer(0.001)

        with tf.name_scope('Gradient_Update'):
            global_varis = self.get_global_variable()
            if self._num_gpus > 1:
                apply_list = []
                for g_v in zip(*gradients_list, global_varis):
                    grads = g_v[:self._num_gpus]
                    varis = g_v[self._num_gpus]
                    with tf.device(varis.device):
                        average_grad = tf.multiply(tf.add_n(grads), 1.0 / self._num_gpus)
                        apply = optimizer.apply_gradients([(average_grad, varis)])
                        apply_list.append(apply)
                with tf.device(global_step.device):
                    apply_list.append(global_step.assign_add(1))
                train_op = tf.group(apply_list)
            else:
                grads_and_varis = list(zip(gradients_list[0], global_varis))
                train_op = optimizer.apply_gradients(grads_and_varis, global_step)

        return train_op
