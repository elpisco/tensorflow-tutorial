import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("data/", one_hot=True)

# Limit data
Xtr, Ytr = mnist.train.next_batch(5000)
Xte, Yte = mnist.test.next_batch(200)

# print Xtr[0, :]
# print Ytr[0, :]

# Graph input
xtr = tf.placeholder(tf.float32, [None, 784])
xte = tf.placeholder(tf.float32, [784])

# Nearest neighbor calculation using L1 distance
# Calculate L1 Distance
# distance = tf.reduce_sum(tf.abs(tf.sub(xtr, xte)), reduction_indices=1)

# Calculate L2 Distance
distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(xtr, xte), 2), reduction_indices=1))

# Prediction: Get min distance index (Nearest neighbor)
pred = tf.arg_min(distance, 0)

accuracy = 0.

# Init variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    # Loop over data
    for i in range(len(Xte)):
        # Get nearest neighbor
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
        # Get class and compare to true label
        print 'Test', i, 'Prediction: ', np.argmax(Ytr[nn_index]), 'True Class: ', np.argmax(Yte[i])
        # Calculate accuracy
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1./len(Xte)
    print 'Done'
    print 'Accuracy: ', accuracy
