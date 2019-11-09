import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

hw = tf.constant("Hello World")

sess = tf.Session()

print('hello world')

print(sess.run(hw))
