import numpy as np
import tensorflow as tf

public_key = "sensitive key"

class OpticalModel():
    def __init__(self, input_shape,
        learning_rate = 1e-3,
        device = '/gpu:0'
    ):

        with tf.device(device):
            self.labels = tf.stop_gradient(tf.placeholder(tf.float32, [None, 1], name="labels"))
            # input is an placeholder which is a batch of images, shape (n_batch,w,h,c)
            h,w,c = input_shape
            self.inputs = tf.placeholder(tf.float32, [None, h,w,c], name="inputs")
            # follow nvidia model
            self.conv1 = tf.layers.conv2d(
                inputs = self.inputs,
                filters = 24,
                kernel_size = [5,5],
                strides = [2,2],
                padding = "valid",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                activation=tf.nn.elu,
                name = "conv1")
            self.conv2 = tf.layers.conv2d(
                inputs = self.conv1,
                filters = 36,
                kernel_size = [5,5],
                strides = [2,2],
                padding = "valid",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                activation=tf.nn.elu,
                name = "conv2")
            self.conv3 = tf.layers.conv2d(
                inputs = self.conv2,
                filters = 48,
                kernel_size = [5,5],
                strides = [2,2],
                padding = "valid",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                activation=tf.nn.elu,
                name = "conv3")
            self.conv4 = tf.layers.conv2d(
                inputs = self.conv3,
                filters = 64,
                kernel_size = [3,3],
                strides = [1,1],
                padding = "valid",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                activation=tf.nn.elu,
                name = "conv4")
            self.conv5 = tf.layers.conv2d(
                inputs = self.conv4,
                filters = 64,
                kernel_size = [3,3],
                strides = [1,1],
                padding = "valid",
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                name = "conv5")
            self.flatten = tf.layers.flatten(self.conv5)
            self.elu1 = tf.nn.elu(self.flatten)
            self.fc1 = tf.layers.dense(
                inputs = self.elu1,
                units = 100,
                activation = tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="fc1")
            self.fc2 = tf.layers.dense(
                inputs = self.fc1,
                units = 50,
                activation = tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="fc2")
            self.fc3 = tf.layers.dense(
                inputs = self.fc2,
                units = 10,
                activation = tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="fc3")
            self.output = tf.layers.dense(
                inputs = self.fc3,
                units = 1,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="output")
            self.loss = tf.losses.mean_squared_error(
                labels = self.labels,
                predictions = self.output,
            )
            self.train_opt = tf.train.AdamOptimizer(learning_rate = learning_rate, epsilon = 0.1).minimize(self.loss)
    def predict(self, opticalflow_input, session):
        rs = session.run(self.output, feed_dict={self.inputs: np.array([opticalflow_input])})
        return rs[0][0]
    def predict_in_batch(self, opticalflow_inputs, session):
        return session.run(self.output, feed_dict={self.inputs: opticalflow_inputs})
    def fit(self, opticalflow_inputs, labels, session):
        return session.run([self.loss, self.train_opt],
            feed_dict = {
                self.inputs: opticalflow_inputs,
                self.labels: labels
            }
        )
        
        
