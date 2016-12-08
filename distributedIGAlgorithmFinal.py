import string
import sys
import math
import glob
import tensorflow as tf

#============================================================
#============================================================
#  This file sits on JOHN in the /tmp/mmays/informationGain/toRemote
#  folder. It is what the code runs on through the distributed tasks file
#============================================================
#============================================================

TRAIN_FILE_PATH = "/tmp/mmays/informationGain/" #sys.argv[0]
#TRAIN_FILE_PATH = "/media/napster/data/train/"
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

        w_h = self.init_weights([self.iSize, 15]) # create symbolic variables
        w_o = self.init_weights([15, self.oSize])

        py_x = self.model(X, w_h, w_o)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) # compute costs
        train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
        predict_op = tf.argmax(py_x, 1)

        # Launch the graph in a session
        with tf.Session() as sess:
            # you need to initialize all variables
            tf.initialize_all_variables().run()

            for i in range(30):
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

def calcSetEntropy(someSet):
    classDictionary = getMalClasses(CLASS_Files)

    classes = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for fileStub in someSet:
        malwareClass = int(classDictionary[fileStub]) - 1
        classes[malwareClass] += 1

    entropy = 0
    for item in range(len(classes)):
        prob = (1.0*classes[item])/len(someSet)
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
                prob = (1.0*item)/len(guessedClasses[guessedClass])
                entropy -= prob * math.log(prob, 2)

        weightedAverageEntropy += entropy * (len(guessedClasses[guessedClass])/len(newClassification))


    return weightedAverageEntropy


def calcInformationGain(origEntropy, newClassification):
    afterEntropy = avgEntropyPerItem(newClassification)
    return origEntropy-afterEntropy


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



def runProcessBacklogItem(batch, titleNum):
    #a full batch of ngrams have been selected, go through files and run these ngrams through FFNN

    batch_data_sets = []
    for subBatch in batch:
		batch_data_sets.append([])
    
    classMatrix = []

    classMatrixBase = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    classDictionary = getMalClasses(CLASS_Files)

    mySet = createEvenTestSet(classDictionary)
    origEntropy = calcSetEntropy(mySet)

    print("Length set: " + str(len(mySet)))

    for fileStubForNGram in mySet:
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

        for subBatch in range(len(batch)):
            myBatchMatrix = []
            for batchItem in batch[subBatch]:
                if batchItem in finderNGramDict:
                    myBatchMatrix.append(finderNGramDict[batchItem])
                else:
                    myBatchMatrix.append(0)

            batch_data_sets[subBatch].append(myBatchMatrix)

    #combined = list(zip(dataMatrix, classMatrix, ASM_Files))
    #random.shuffle(combined)

    #dataMatrix[:], classMatrix[:], file_list = zip(*combined)
    #file_list = ASM_Files[:]

    returnData = []
    for subBatch in range(len(batch)):
        myNet = FFNN(batch_data_sets[subBatch], classMatrix, [], [], len(batch[subBatch]), 9)
        result = myNet.trainAndClassify()
        findEntropySet = []
        for item in range(len(result)):
            findEntropySet.append([mySet[item], result[item]])

        informationGainDiff = calcInformationGain(origEntropy, findEntropySet)
        returnData.append([batch[subBatch], informationGainDiff])

    uniqueID = str(titleNum)
    resultFile = open(TRAIN_FILE_PATH + "results/" + uniqueID, "a")
    for item in returnData:
        resultFile.write(str(item) + "\n")

batches = list(eval(sys.argv[1]))
tmpTaskFile = open(TRAIN_FILE_PATH + "TaskLog.txt", 'r')
vals = tmpTaskFile.readlines()
tmpTaskFile.close()

processVals = []
for batch in batches:
    processVals.append(list(eval(vals[batch])))
    
runProcessBacklogItem(processVals, batches[0])
