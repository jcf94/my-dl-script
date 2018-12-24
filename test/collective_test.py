import tensorflow as tf
from tensorflow.python.ops import collective_ops
from tensorflow.python.framework import device as pydev

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

CONFIG = tf.ConfigProto()
#CONFIG.log_device_placement=True

with tf.device('cpu:0'):
    v0 = tf.Variable([1, 1, 1], dtype=tf.float32)

with tf.device('gpu:0'):
    v1 = tf.Variable([2, 2, 2], dtype=tf.float32)

with tf.device('gpu:1'):
    v2 = tf.Variable([3, 3, 3], dtype=tf.float32)
    v3 = tf.Variable([5, 5, 5], dtype=tf.float32)

sess = tf.Session(config=CONFIG)

sess.run(tf.global_variables_initializer())

print(sess.run([v0, v1, v2]))

sum_reduce = []
# with tf.device('cpu:0'):
#     out.append(collective_ops.all_reduce(v0, 3, 1, 1, 'Add', 'Id'))
with tf.device('gpu:0'):
    sum_reduce.append(collective_ops.all_reduce(v1, 2, 0, 1, 'Add', 'Id'))
with tf.device('gpu:1'):
    sum_reduce.append(collective_ops.all_reduce(v2, 2, 0, 1, 'Add', 'Id'))
print(sess.run(sum_reduce))

average_reduce = []
with tf.device('gpu:0'):
    average_reduce.append(collective_ops.all_reduce(v1, 2, 1, 1, 'Add', 'Div'))
with tf.device('gpu:1'):
    average_reduce.append(collective_ops.all_reduce(v2, 2, 1, 1, 'Add', 'Div'))
print(sess.run(average_reduce))

print('==========================')

bcast = []
# with tf.device('cpu:0'):
#     bcast.append(collective_ops.broadcast_send(v0, v0.shape, v0.dtype, 2, 3, 1))
with tf.device('gpu:0'):
    bcast.append(collective_ops.broadcast_send(v1, v1.shape, v1.dtype, 2, 3, 2))
with tf.device('gpu:1'):
    bcast.append(collective_ops.broadcast_recv(v1.shape, v1.dtype, 2, 3, 2))

print(sess.run(bcast))

print('==========================')

average_reduce = []
with tf.device('gpu:0'):
    average_reduce.append(collective_ops.all_reduce(v1, 2, 4, 3, 'Add', 'Div'))
with tf.device('gpu:1'):
    average_reduce.append(collective_ops.all_reduce(v3, 2, 4, 3, 'Add', 'Div'))
print(sess.run(average_reduce))