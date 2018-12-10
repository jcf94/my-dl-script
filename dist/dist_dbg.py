import os
import time
import math
from datetime import datetime
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

PID = os.getpid()
print('Program pid:', PID)
print('Pause here to enter DBG')
os.system('GREPDB="read"; /bin/bash -c "$GREPDB"')

CONFIG = tf.ConfigProto()
CONFIG.gpu_options.allow_growth=True
CONFIG.log_device_placement=True

tf.app.flags.DEFINE_string("ps_hosts", "localhost:50000",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "localhost:50001",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS

def simple_test(cluster, server):

    with tf.device("/job:ps/task:0"):
        a = tf.Variable(tf.ones([10]))

    with tf.device("/job:worker/task:0"):
        b = tf.Variable(tf.ones([10]))
        c = b.assign_add(a);
    
    init_op = tf.initialize_all_variables()

    with tf.Session(server.target, config=CONFIG) as sess:

        sess.run(init_op)
        print(sess.run(c))

def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    if FLAGS.job_name == "ps":
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif FLAGS.job_name == "worker":
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    server = tf.train.Server(cluster,
                            job_name=FLAGS.job_name,
                            task_index=FLAGS.task_index, protocol="grpc+verbs")

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        program_start_time = time.time()
        simple_test(cluster, server)
        program_end_time = time.time()
        print('Program finished, Total seconds: %s' % (program_end_time - program_start_time))

if __name__ == '__main__':
    tf.app.run()
