import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def _parse_example(serialized_example):
    """Return inputs and targets Tensors from a serialized tf.Example."""
    data_fields = {
        "inputs": tf.VarLenFeature(tf.int64),
        "targets": tf.VarLenFeature(tf.int64)
    }
    parsed = tf.parse_single_example(serialized_example, data_fields)
    inputs = tf.sparse_tensor_to_dense(parsed["inputs"])
    targets = tf.sparse_tensor_to_dense(parsed["targets"])
    return inputs, targets

filename = ["wmt14_data/wmt32k-train-00001-of-00100"]
dataset = tf.data.TFRecordDataset(filename)

dataset = dataset.map(_parse_example)
iterator = dataset.make_initializable_iterator()

sess = tf.Session()
sess.run(iterator.initializer)

for i in range(10):
    out = sess.run(iterator.get_next())
    print(out[0].shape, " ", out[1].shape)