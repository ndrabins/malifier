import numpy as np
import math
import glob
import tensorflow as tf

TRAIN_FILE_PATH = "/media/napster/data/train/"
MALWARE_FILE_PATH = TRAIN_FILE_PATH + "nGramFeatures/"

ASM_Files = glob.glob(MALWARE_FILE_PATH + "*.txt")
CLASS_Files = TRAIN_FILE_PATH + "trainLabels.txt"


class FFNN:
    def __init__(self, trainData, trainLabels, testData, testLabels, inputSize, outputSize):
        self.trX = trainData
        self.trY = trainLabels
        self.teX = testData
        self.teY = testLabels
        self.iSize = inputSize
        self.oSize = outputSize

        self.X = tf.placeholder("float", [None, self.iSize])
        self.Y = tf.placeholder("float", [None, self.oSize])

        # !!!!!!!!!!!!!!!!
        #I probably want nowhere near 625 for finding information gain --- see if changes need to be made
        # !!!!!!!!!!!!!!!!

        self.w_h = self.init_weights([self.iSize, int((self.iSize*2)/3)]) # create symbolic variables
        self.w_o = self.init_weights([int((self.iSize*2)/3), self.oSize])

        self.py_x = self.model(self.X, self.w_h, self.w_o)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.py_x, self.Y)) # compute costs
        self.train_op = tf.train.GradientDescentOptimizer(0.05).minimize(self.cost) # construct an optimizer
        self.predict_op = tf.argmax(self.py_x, 1)


    def init_weights(self, shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.01))


    def model(self, X, w_h, w_o):
        h = tf.nn.sigmoid(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
        return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us


    def train(self):
        # Launch the graph in a session
        with tf.Session() as sess:
            # you need to initialize all variables
            tf.initialize_all_variables().run()

            for i in range(5000):
                if i % 1000 == 0:
                    print("Training iteration --| " + str(i))
                #run training
                for batch in range(0, len(self.trX) - 50, 50):
                    sess.run(self.train_op, feed_dict={self.X: self.trX[batch:batch+50], self.Y: self.trY[batch:batch+50]})

    def classify(self):
        with tf.Session() as sess:
            #show test results
            #print(i, np.mean(np.argmax(self.teY, axis=1) ==
            #                 sess.run(predict_op, feed_dict={X: self.teX})))
            value = sess.run(self.predict_op, feed_dict={self.X: self.teX})
            return value

def getMalClasses(fName):
    classDictionary = {}
    file = open(fName, 'r')
    for line in file:
        nameAndClass = line.split()
        classDictionary[nameAndClass[0]] = nameAndClass[1]

    return classDictionary

def createEvenTestSet(classDictionary):
    classesSet = []

    classVals = [1,2,3,4,5,6,7,8,9]
    min = len(ASM_Files)
    counterHolder = []
    for counter in range(len(classVals)):
        counterHolder.append(0)
        num = list(classDictionary.values()).count(str(classVals[counter]))
        if num < min:
            min = num

    for item in range(len(ASM_Files)):
        if counterHolder[classVals.index(int(classDictionary[ASM_Files[item].split('/')[-1][:-4]]))] < min:
            classesSet.append(ASM_Files[item].split('/')[-1][:-4])
            counterHolder[classVals.index(int(classDictionary[ASM_Files[item].split('/')[-1][:-4]]))] += 1

    return classesSet

def calcAccuracy(newClass, trueClass):

    correct = 0
    for item in range(len(newClass)):
        if newClass[item][trueClass[item]] == 1:
            correct += 1

    return correct/len(newClass)

def runClassifier(featureBatch):
    #a full batch of ngrams have been selected, go through files and run these ngrams through FFNN

    classMatrix = []

    classMatrixBase = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    classDictionary = getMalClasses(CLASS_Files)

    mySet = createEvenTestSet(classDictionary)

    test_data_matrix = []
    test_class_matrix = []


    data_matrix = []
    for fileStubForNGram in mySet:
        temp = classMatrixBase[:]
        temp[int(classDictionary[fileStubForNGram]) - 1] = 1
        classMatrix.append(temp)

        try:
            file = open(TRAIN_FILE_PATH + "nGramFeatures/" + fileStubForNGram + ".txt", 'r')
            finderNGramDict = dict(eval(file.readline()))
            file.close()
        except FileNotFoundError:
            # for some reason there are 2 (so far)
            continue

        myBatchMatrix = []
        for item in range(len(featureBatch)):
            if featureBatch[item] in finderNGramDict:
                myBatchMatrix.append(finderNGramDict[featureBatch[item]])
            else:
                myBatchMatrix.append(0)

        data_matrix.append(myBatchMatrix)


    for file in ASM_Files:
        fileStubForNGram = file.split('/')[-1][:-4]
        temp = classMatrixBase[:]
        temp[int(classDictionary[fileStubForNGram]) - 1] = 1
        test_class_matrix.append(temp)

        try:
            file = open(TRAIN_FILE_PATH + "nGramFeatures/" + fileStubForNGram + ".txt", 'r')
            finderNGramDict = dict(eval(file.readline()))
            file.close()
        except FileNotFoundError:
            # for some reason there are 2 (so far)
            continue

        myBatchMatrix = []
        for item in range(len(featureBatch)):
            if featureBatch[item] in finderNGramDict:
                myBatchMatrix.append(finderNGramDict[featureBatch[item]])
            else:
                myBatchMatrix.append(0)

        test_data_matrix.append(myBatchMatrix)

    # random.shuffle(combined)

    # dataMatrix[:], classMatrix[:], file_list = zip(*combined)
    #file_list = ASM_Files[:]

    returnData = []

    myNet = FFNN(data_matrix, classMatrix, test_data_matrix, test_class_matrix, len(featureBatch), 9)
    myNet.train()
    result = myNet.classify()

    #findEntropySet = []
    #for item in range(len(result)):
    #    findEntropySet.append([mySet[item], result[item]])

    accuracy = calcAccuracy(test_data_matrix, result)
    return accuracy

def doFullRun():
    myFile = open("/media/napster/data/train/informationGain/aboveZeroNew.txt", 'r')
    allFeatures = myFile.readlines()
    featureBatch = []
    for feature in allFeatures:
        featureBatch.append(list(eval(feature))[0])


    accuracy = runClassifier(featureBatch)
    print(accuracy)

doFullRun()