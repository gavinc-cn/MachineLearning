import tensorflow as tf
import os
import pickle
import numpy as np


CIFAR_DIR = "../data/cifar-10-batches-py"
print(os.listdir(CIFAR_DIR))


def load_data(filename):
    """read data from data file."""
    with open(filename, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()
        return data['data'], data['labels']


class CifarData:

    def __init__(self, filenames, need_shuffle):
        pass

x = tf.placeholder(tf.float32, [None, 3072])
y = tf.placeholder(tf.int64, [None])

# (3072,  1)
w = tf.get_variable('w', [x.get_shape()[-1], 1], initializer=tf.random_normal_initializer(0, 1))

# (1, )
b = tf.get_variable('b', [1], initializer=tf.constant_initializer(0.0))

# [None, 3072] * [3072, 1] = [None, 1]
y_ = tf.matmul(x, w) + b

# [None, 1]
p_y_1 = tf.nn.sigmoid(y_)
# [None, 1]
y_reshaped = tf.reshape(y, (-1, 1))
y_reshaped_float = tf.cast(y_reshaped, tf.float32)

loss = tf.reduce_mean(tf.square(y_reshaped_float - p_y_1))

# bool
predict = p_y_1 > 0.5
# [1, 0, 1, 1, 1, 0, 0]
correct_prediction = tf.equal(tf.cast(predict, tf.int64), y_reshaped)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

init = tf.global_variable_initializer()
with tf.Session() as sess:
    sess.run([loss, accuracy, train_op], feed_dict={})



