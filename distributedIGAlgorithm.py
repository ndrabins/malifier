import sys
import math
import glob
import tensorflow as tf

TRAIN_FILE_PATH = sys.argv[0]
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


def getMalClasses(fName):
    classDictionary = {}
    file = open(fName, 'r')
    for line in file:
        nameAndClass = line.split()
        classDictionary[nameAndClass[0]] = nameAndClass[1]

    return classDictionary

def calcSetEntropy():
    classDictionary = getMalClasses(CLASS_Files)

    classes = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for file in ASM_Files:
        fileStub = file.split('/')[-1][:-4]  # grab all but last chars in file name (removes .asm)
        malwareClass = int(classDictionary[fileStub]) - 1
        classes[malwareClass] += 1

    entropy = 0
    for item in range(len(classes)):
        prob = classes[item]/len(CLASS_Files)
        entropy -= prob * math.log(prob, 2)  # 3.0

    return entropy

def avgEntropyPerItem(newClassification):
    ####################################################################################
    # See http://www.math.unipd.it/~aiolli/corsi/0708/IR/Lez12.pdf for help
    ####################################################################################

    #newClassification needs to be in format -> [[fileStub, guessedClass],[fileStub, guessedClass]]
    guessedClasses = [[],[],[],[],[],[],[],[],[]]
    for guess in newClassification:
        guessedClasses[guess[1]].append(guess[0])

    #guessedClasses looks like this [[fileStub,fileStub],[fileStub,fileStub]] for each class


    ############################################
    #setup for comparison
    ############################################



    classDictionary = getMalClasses(CLASS_Files)

    ############################################
    ############################################

    weightedAverageEntropy = 0
    for guessedClass in range(len(guessedClasses)):
        #### Calc intermediary Entropy per class ####

        classes = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        for fileStub in guessedClasses[guessedClass]:
            trueClass = int(classDictionary[fileStub]) - 1
            classes[trueClass] += 1

        entropy = 0
        for item in classes:
            if item > 0:
                prob = item/len(guessedClasses[guessedClass])
                entropy -= prob * math.log(prob, 2)

        weightedAverageEntropy += entropy * (len(guessedClasses[guessedClass])/len(newClassification))


    return weightedAverageEntropy


def calcInformationGain(origEntropy, newClassification):
    afterEntropy = avgEntropyPerItem(newClassification)
    return origEntropy-afterEntropy


def runProcessBacklogItem(batch):
    #a full batch of ngrams have been selected, go through files and run these ngrams through FFNN

    dataMatrix = []
    classMatrix = []

    classMatrixBase = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    classDictionary = getMalClasses(CLASS_Files)

    origEntropy = calcSetEntropy()

    for fNameForNGram in ASM_Files:
        fileStubForNGram = fNameForNGram.split('/')[-1][:-4]
        temp = classMatrixBase[:]
        temp[int(classDictionary[fileStubForNGram])-1] = 1
        classMatrix.append(temp)

        try:
            file = open(TRAIN_FILE_PATH + "nGramFeatures/" + fileStubForNGram + ".txt", 'r')
            finderNGramDict = dict(eval(file.readline()))
            file.close()
        except FileNotFoundError:
            #for some reason there are 2 (so far)
            continue

        myBatchMatrix = []

        for batchItem in batch:
            if batchItem in finderNGramDict:
                myBatchMatrix.append(finderNGramDict[batchItem])
            else:
                myBatchMatrix.append(0)

        dataMatrix.append(myBatchMatrix)

    #combined = list(zip(dataMatrix, classMatrix, ASM_Files))
    #random.shuffle(combined)

    #dataMatrix[:], classMatrix[:], file_list = zip(*combined)
    file_list = ASM_Files[:]
    myNet = FFNN(dataMatrix, classMatrix, [], [], len(batch), 9)
    result = myNet.trainAndClassify()
    findEntropySet = []
    for item in range(len(result)):
        findEntropySet.append([file_list[item].split('/')[-1][:-4], result[item]])

    #the findEntropySet variable is different than what is expected in the function....will need to change the function
    informationGainDiff = calcInformationGain(origEntropy, findEntropySet)

    #myIGFile.write("[" + str(batch) + ", " + str(informationGainDiff) + "]")

    #myIGFile.close()

runProcessBacklogItem(list(eval(sys.argv[0])))