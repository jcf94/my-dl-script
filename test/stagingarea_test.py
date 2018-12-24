import numpy as np
import tensorflow as tf
from tensorflow.python.ops.data_flow_ops import StagingArea
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

length = 10
dataset_range = tf.data.Dataset.range(length)
iter = dataset_range.make_one_shot_iterator()
next_item = iter.get_next()

with tf.device('cpu:0'):
    a = tf.Variable([1], dtype=tf.int64)

    area = StagingArea(dtypes=[tf.int64])
    area_put = area.put([next_item])
    area_get = area.get()[0]
    area_size = area.size()
    area_get_put = tf.tuple([area_get], control_inputs=[area_put])[0]

    b = a + area_get
    c = b + area_get

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # first item, just put()
    print('put:', sess.run(area_put))
    # get() & put()
    for i in range(length - 1):
        # this works as "semicolon"
        print(sess.run(c))
        print('put(); get() =', sess.run(area_put))
        print('size:', sess.run(area_size))
    # last item, just get()
    print('get() =', sess.run(area_get))
    print('size:', sess.run(area_size))