import tensorflow as tf

conf = tf.ConfigProto(
	device_count = {'GPU': 0}
)

a = tf.constant(1)
b = tf.constant(2)
c = a + b

sess = tf.Session(config=conf)

sess.run(tf.global_variables_initializer())

print(sess.run(c))