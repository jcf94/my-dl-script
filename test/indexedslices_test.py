import tensorflow as tf

## 10-Dimensional data will be fed to the model
X = tf.Variable( tf.ones([10, 5]) )

## W works with the first 3 features of a sample
W = tf.Variable( tf.ones( [5, 3] ) )
Xi = tf.gather( X, [0, 1, 3] )

Y = tf.Variable( tf.ones([10, 5]) )

Yi = tf.gather( Y, [1, 3, 6] )

print(Xi)

mm = tf.matmul( W, Xi ) + tf.matmul( W, Yi)

gX, gY = tf.gradients(mm, [X, Y])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(gX)
print(gY)

g = [gX, gY]

print(sess.run(tf.multiply(tf.add_n(g), 1/5.0)))