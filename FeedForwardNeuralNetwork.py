###############################################
###############################################3
# Taken from https://github.com/nlintz/TensorFlow-Tutorials
###############################################3
###############################################

import tensorflow as tf
import numpy as np


class FFNN:
    def __init__(self, trainData, trainLabels, testData, testLabels, inputSize, outputSize):
        self.trX = trainData
        self.trY = trainLabels
        self.teX = testData
        self.teY = testLabels
        self.iSize = inputSize
        self.oSize = outputSize


    def init_weights(self, shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.01))


    def model(self, X, w_h, w_o):
        h = tf.nn.sigmoid(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
        return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us


    def trainAndClassify(self):
        X = tf.placeholder("float", [None, self.iSize])
        Y = tf.placeholder("float", [None, self.oSize])

        # !!!!!!!!!!!!!!!!
        #I probably want nowhere near 625 for finding information gain --- see if changes need to be made
        # !!!!!!!!!!!!!!!!

        w_h = self.init_weights([self.iSize, 625]) # create symbolic variables
        w_o = self.init_weights([625, self.oSize])

        py_x = self.model(X, w_h, w_o)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) # compute costs
        train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
        predict_op = tf.argmax(py_x, 1)

        # Launch the graph in a session
        with tf.Session() as sess:
            # you need to initialize all variables
            tf.initialize_all_variables().run()

            for i in range(500):
                #run training
                for batch in range(0, len(self.trX) - 50, 50):
                    sess.run(train_op, feed_dict={X: self.trX[batch:batch+50], Y: self.trY[batch:batch+50]})


            #show test results
            #print(i, np.mean(np.argmax(self.teY, axis=1) ==
            #                 sess.run(predict_op, feed_dict={X: self.teX})))
            value = sess.run(predict_op, feed_dict={X: self.trX})
            return value