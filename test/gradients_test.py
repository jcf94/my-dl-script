import tensorflow as tf 

a = tf.constant(0.)
b = 2 * a
g = tf.gradients(a + b, [a, b])

print(g)