import tensorflow as tf
import numpy as np

#====================================================================================
#====================================================================================
#           A convolutional neural network programmed using tensorflow
#
#====================================================================================
#====================================================================================

class TFConvNetwork:

    def __init__(self, iWidth, iHeight, outputSize, trainData, trainLabels, testData, testLabels):
        tf.reset_default_graph()
        self.imageWidth = iWidth
        self.imageHeight = iHeight

        # the 1600 is the number of pixels in an image and the 10 is the number of images in a batch
        # ...aka for labels
        self.X = tf.placeholder(tf.float32, shape=[None, iHeight, iWidth, 1])
        self.Y = tf.placeholder(tf.float32, shape=[None, outputSize])

        self.trX = np.asarray(trainData).reshape(-1, iHeight, iWidth, 1)
        self.trY = trainLabels

        self.teX = np.asarray(testData).reshape(-1, iHeight, iWidth, 1)
        self.teY = testLabels

        w = self.init_weights([3, 3, 1, 32])  # 3x3x1 conv, 32 outputs
        w2 = self.init_weights([3, 3, 32, 64])  # 3x3x32 conv, 64 outputs
        w3 = self.init_weights([3, 3, 64, 128])  # 3x3x32 conv, 128 outputs
        w4 = self.init_weights([128 * 4 * 4, 625])  # FC 128 * 4 * 4 inputs, 625 outputs
        w_o = self.init_weights([625, outputSize])  # FC 625 inputs, 10 outputs (labels)

        self.p_keep_conv = tf.placeholder("float")
        self.p_keep_hidden = tf.placeholder("float")

        self.py_x = self.model(self.X, w, w2, w3, w4, w_o, self.p_keep_conv, self.p_keep_hidden)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.py_x, self.Y))
        self.train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(self.cost)
        self.predict_op = tf.argmax(self.py_x, 1)

        self.loose_predict_op = self.py_x


    def model(self, X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
        l1a = tf.nn.relu(tf.nn.conv2d(X, w,  # l1a shape=(?, 28, 28, 32)
                                      strides=[1, 1, 1, 1], padding='SAME'))
        l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],  # l1 shape=(?, 14, 14, 32)
                            strides=[1, 2, 2, 1], padding='SAME')
        l1 = tf.nn.dropout(l1, p_keep_conv)

        l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,  # l2a shape=(?, 14, 14, 64)
                                      strides=[1, 1, 1, 1], padding='SAME'))
        l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],  # l2 shape=(?, 7, 7, 64)
                            strides=[1, 2, 2, 1], padding='SAME')
        l2 = tf.nn.dropout(l2, p_keep_conv)

        l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,  # l3a shape=(?, 7, 7, 128)
                                      strides=[1, 1, 1, 1], padding='SAME'))
        l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],  # l3 shape=(?, 4, 4, 128)
                            strides=[1, 2, 2, 1], padding='SAME')
        l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])  # reshape to (?, 2048)
        l3 = tf.nn.dropout(l3, p_keep_conv)

        l4 = tf.nn.relu(tf.matmul(l3, w4))
        l4 = tf.nn.dropout(l4, p_keep_hidden)

        pyx = tf.matmul(l4, w_o)
        return pyx

    #============================================================================
#============================================================================
#                       This runs the code
#============================================================================
#============================================================================
    def init_weights(self, shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.01))

    def trainAndClassify(self, iter):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            print("         --------    Train  |  Test")
            for i in range(iter):
                for batch in range(0, len(self.trX) - 50, 50):
                    sess.run(self.train_op, feed_dict={self.X: self.trX[batch:batch+50], self.Y: self.trY[batch:batch+50],
                                                      self.p_keep_conv: 0.8, self.p_keep_hidden: 0.5})

                if i % 1 == 0:
                    trainVal = round(np.mean(np.argmax(self.trY, axis=1) ==
                                      sess.run(self.predict_op, feed_dict={self.X: self.trX,
                                                                                self.p_keep_conv: 1.0,
                                                                                self.p_keep_hidden: 1.0})),4)

                    testVal =  round(np.mean(np.argmax(self.teY, axis=1) ==
                                                  sess.run(self.predict_op, feed_dict={self.X: self.teX,
                                                                                       self.p_keep_conv: 1.0,
                                                                                       self.p_keep_hidden: 1.0})),4)


                    print("           ", str(i).ljust(7), str(trainVal).ljust(6), " | ", str(testVal).ljust(6))
            value=  np.mean(np.argmax(self.teY, axis=1) ==
                                                  sess.run(self.predict_op, feed_dict={self.X: self.teX,
                                                                                       self.p_keep_conv: 1.0,
                                                                                       self.p_keep_hidden: 1.0}))

            value2 = sess.run(self.loose_predict_op, feed_dict={self.X: self.teX, self.p_keep_conv: 1.0, self.p_keep_hidden: 1.0})
            return [value2, value]



#HELPER FUNCTIONS
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')