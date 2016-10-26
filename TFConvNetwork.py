import tensorflow as tf
import numpy


class TFConvNetwork:

    def __init__(self, iWidth, iHeight):
        self.imageWidth = iWidth
        self.imageHeight = iHeight
        self.sess = tf.InteractiveSession()

        # the 1600 is the number of pixels in an image and the 10 is the number of images in a batch
        # ...aka for labels
        self.x = tf.placeholder(tf.float32, shape=[None, iWidth*iHeight])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 9])

        # ============================================================================
        # ============================================================================
        #                       FIRST CONVOLUTIONAL LAYER
        # ============================================================================
        # ============================================================================

        # 5x5 patch with 1 input and 32 output features
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])

        # 28x28 input image with 1 color layer (this will remain one since it is byte "image")
        x_image = tf.reshape(self.x, [-1, self.imageWidth, self.imageHeight, 1])

        # perform convolution on x_image and W_conv1 and add bias before doing ReLU)
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

        # perform max pooling
        h_pool1 = max_pool_2x2(h_conv1)

        # ============================================================================
        # ============================================================================
        #                       SECOND CONVOLUTIONAL LAYER
        # ============================================================================
        # ============================================================================

        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        currSize = int((iWidth/2)/2)
        currSize2 = int((iHeight / 2) / 2)

        W_fc1 = weight_variable([currSize * currSize2 * 64, 1024])
        b_fc1 = bias_variable([1024])

        # ============================================================================
        # ============================================================================
        #                       DENSELY CONNECTED LAYER
        # ============================================================================
        # ============================================================================

        h_pool2_flat = tf.reshape(h_pool2, [-1, currSize * currSize2 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # ============================================================================
        # ============================================================================
        #                       DROPOUT (To avoid overfitting)
        # ============================================================================
        # ============================================================================

        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # ============================================================================
        # ============================================================================
        #                       READOUT (with softmax)
        # ============================================================================
        # ============================================================================

        W_fc2 = weight_variable([1024, 9])
        b_fc2 = bias_variable([9])

        self.y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#============================================================================
#============================================================================
#                       This runs the code
#============================================================================
#============================================================================

    def train(self, trainSet, iter):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_conv), reduction_indices=[1]))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.sess.run(tf.initialize_all_variables())

        #20000 iterations
        spot = 0
        spot2 = 0
        step = 50
        for i in range(iter):
            spot2 = spot + 50
            if (spot2 >= len(trainSet[0])):
                spot2 = spot2 % len(trainSet[0])

                data = numpy.asarray(trainSet[0][spot:len(trainSet[0])-1] + trainSet[0][0:spot2])
                labels = numpy.asarray(trainSet[1][spot:len(trainSet[1]) - 1] + trainSet[1][0:spot2])
            else:
                data = numpy.asarray(trainSet[0][spot:spot2])
                labels = numpy.asarray(trainSet[1][spot:spot2])
                #labels = numpy.reshape(labels, 9, 1)

            batch = (data, labels)
            spot = spot2

            if i%100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    self.x:batch[0], self.y_: batch[1], self.keep_prob: 1.0})
                print("step %d, training accuracy %g"%(i, train_accuracy))
                train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})

        #print("test accuracy %g"%accuracy.eval(feed_dict={
        #    self.x: mnist.test.images, self.y_: mnist.test.labels, self.keep_prob: 1.0}))

    def test(self, testSet):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_conv), reduction_indices=[1]))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.sess.run(tf.initialize_all_variables())
        for i in range((len(testSet)/50)-1):
            data = numpy.asarray(testSet[0][i*50:(i*50)+50])
            labels = numpy.asarray(testSet[1][i*50:(i*50)+50])

        batch = (data, labels)
        print("test accuracy %g" % accuracy.eval(feed_dict={
            self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0}))

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