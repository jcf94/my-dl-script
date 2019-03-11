import tensorflow as tf
import os
import record_data_read as rdread

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Record Data Structure
# {
#     'image/height': _int64_feature(height),
#     'image/width': _int64_feature(width),
#     'image/colorspace': _bytes_feature(colorspace),
#     'image/channels': _int64_feature(channels),
#     'image/class/label': _int64_feature(label),
#     'image/class/synset': _bytes_feature(synset),
#     'image/class/text': _bytes_feature(human),
#     'image/object/bbox/xmin': _float_feature(xmin),
#     'image/object/bbox/xmax': _float_feature(xmax),
#     'image/object/bbox/ymin': _float_feature(ymin),
#     'image/object/bbox/ymax': _float_feature(ymax),
#     'image/object/bbox/label': _int64_feature([label] * len(xmin)),
#     'image/format': _bytes_feature(image_format),
#     'image/filename': _bytes_feature(os.path.basename(filename)),
#     'image/encoded': _bytes_feature(image_buffer)
# }

def map_fn(example_serialized):
    image_buffer, label_index, bbox, _ = rdread.parse_example_proto(example_serialized)
    return rdread.simple_process(image_buffer, bbox, 299, 299, 3, True), label_index
    # return ip.image_preprocessing(image_buffer, bbox, True, 0)

DATA_DIR = "record_data/"
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

data = ImageNet_Data('ImageNet', 'validation')

dataset = data.dataset()

# filename = ["record_data/train-00000-of-01024"]
# dataset = tf.data.TFRecordDataset(filename)
dataset = dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(batch_size=64)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
# dataset = dataset.repeat()

iterator = dataset.make_initializable_iterator()

sess = tf.Session()

count = 0

for i in range(5):
    sess.run(iterator.initializer)
    while (True):
        try:
            sess.run(iterator.get_next())
            count += 1
            print(count)
        except tf.errors.OutOfRangeError:
            print("Done one epoch")
            break;
