###############################################
###############################################3
# Taken from https://github.com/nlintz/TensorFlow-Tutorials
###############################################3
###############################################

import tensorflow as tf
import numpy as np

class FFNN:
    def __init__(self, trainData, trainLabels, testData, testLabels, inputSize, outputSize, iterations):
        tf.reset_default_graph()
        self.trX = trainData
        self.trY = trainLabels
        self.teX = testData
        self.teY = testLabels
        self.iSize = inputSize
        self.oSize = outputSize
        self.it = iterations

        self.X = tf.placeholder("float", [None, self.iSize])
        self.Y = tf.placeholder("float", [None, self.oSize])

        self.w_h = self.init_weights([self.iSize, int((self.iSize*2)/3)]) # create symbolic variables
        self.w_h2 =self.init_weights([int((self.iSize*2)/3), int((self.iSize*2)/3)]) # create symbolic variables
        self.w_o = self.init_weights([int((self.iSize*2)/3), self.oSize])

        self.p_keep_input = tf.placeholder("float")
        self.p_keep_hidden = tf.placeholder("float")

        self.py_x = self.model(self.X, self.w_h, self.w_h2, self.w_o, self.p_keep_input, self.p_keep_hidden)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.py_x, self.Y)) # compute costs
        self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.cost) # construct an optimizer
        #with tf.Session() as sess:
        #    sess.run(tf.initialize_variables([for name in self.train_op.get_slot_names()])
        #self.train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(self.cost)
        self.predict_op = tf.argmax(self.py_x, 1)

        self.loose_predict_op = self.py_x

        self.init_op = tf.initialize_all_variables()

    def init_weights(self, shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.01))


    def model(self, X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden):
        X = tf.nn.dropout(X, p_keep_input)
        h = tf.nn.relu(tf.matmul(X, w_h))

        h = tf.nn.dropout(h, p_keep_hidden)
        h2 = tf.nn.relu(tf.matmul(h, w_h2))

        h2 = tf.nn.dropout(h2, p_keep_hidden)

        return tf.matmul(h2, w_o)

        #h = tf.nn.sigmoid(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
        #h2 = tf.nn.sigmoid(tf.matmul(w_h, w_h2))
        #return tf.matmul(h2, w_o) # note that we dont take the softmax at the end because our cost fn does that for us


    def trainAndClassify(self):
        # Launch the graph in a session
        print("         --------    Train  |  Test")
        with tf.Session() as sess:
            # you need to initialize all variables
            #tf.initialize_all_variables().run()
            sess.run(self.init_op)

            for i in range(self.it):
                #run training
                for batch in range(0, len(self.trX) - 50, 50):
                    sess.run(self.train_op, feed_dict={self.X: self.trX[batch:batch+50], self.Y: self.trY[batch:batch+50], self.p_keep_input: 0.8, self.p_keep_hidden: 0.5})

                trainVal = round(np.mean(np.argmax(self.trY, axis=1) ==
                                 sess.run(self.predict_op, feed_dict={self.X: self.trX,
                                                                 self.p_keep_input: 1.0,
                                                                 self.p_keep_hidden: 1.0})), 4)

                testVal = round(np.mean(np.argmax(self.teY, axis=1) ==
                                 sess.run(self.predict_op, feed_dict={self.X: self.teX,
                                                                 self.p_keep_input: 1.0,
                                                                 self.p_keep_hidden: 1.0})), 4)

                print("           ", str(i).ljust(7), str(trainVal).ljust(6), " | ", str(testVal).ljust(6))

                if testVal > 0.97 and trainVal > 0.97:
                    break

            value = sess.run(self.predict_op, feed_dict={self.X: self.teX, self.p_keep_input: 1.0,
                                                                 self.p_keep_hidden: 1.0})

            value2 = sess.run(self.loose_predict_op, feed_dict={self.X: self.teX, self.p_keep_input: 1.0,
                                                                 self.p_keep_hidden: 1.0})


        correct = 0
        for item in range(len(value)):
            if self.teY[item][value[item]] == 1:
                correct += 1

        return [value2, correct/len(value)]

