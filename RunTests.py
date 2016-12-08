import glob

import numpy as np
import tools
import TFConvNetwork
import classDict
import getFeatures
import numpy
import FeedForwardNeuralNetwork
import distributedIGAlgorithm

TRAIN_FILE_PATH = "/media/napster/data/train/" #"F:/train/"
MALWARE_FILE_PATH = TRAIN_FILE_PATH + "train/"

ASM_Files = glob.glob(MALWARE_FILE_PATH + "*.asm")
CLASS_Files = TRAIN_FILE_PATH + "trainLabels.txt"

NGRAM_FEATURES_FILENAME = "aboveZeroNew.txt"

#CNN SPECS
CNN_ITERATIONS = 50
X_DIMENSION = 16
Y_DIMENSION = 64

#NGRAM SPECS
iterations = 20

#build info dict for getting CNN features
info = {}
info["X"] = X_DIMENSION
info["Y"] = Y_DIMENSION

#build info dict for getting nGram features
nGramFeaturesFile = open(TRAIN_FILE_PATH + "informationGain/" + NGRAM_FEATURES_FILENAME, 'r')
allFeatures = nGramFeaturesFile.readlines()
info["featureBatch"] = []
for feature in allFeatures:
    info["featureBatch"].append(list(eval(feature))[0])

#Get the lookup table for Malware Classes
classDictionary = classDict.getMalClasses(CLASS_Files)

#Create Simple Train/Test partition
mySet = tools.createEvenTestSet(classDictionary, ASM_Files)
#mySet = tools.buildAllSet(ASM_Files)

#Assemble features in train set
#==================================
# This is the section that will take the longest time running since it will actually be extracting the features of
# the malware for each file. This could be a lengthy process.
#==================================
CNN_accuracy = 0
NGRAM_accuracy = 0
combAccPercent = 0
numCrossVals = 10

data_cnn = []
data_nGram = []
base = [0, 0, 0, 0, 0, 0, 0, 0, 0]
labels = []
for fileStub in mySet:
    featureListNGRAM = getFeatures.getFeatures(fileStub, "NGRAM", info)
    featureList = getFeatures.getFeatures(MALWARE_FILE_PATH + fileStub, "CNN", info)

    #check for error
    if (len(featureList) == 0 or len(featureListNGRAM) == 0):
        #print("FEATURE GRAB ERROR")
        #print("Not adding file: " + fileStub + "to the train set")
        continue

    #Add values to data matrices (one matrix per classifier)
    data_nGram.append(numpy.asarray(featureListNGRAM))
    data_cnn.append(numpy.asarray(featureList))

    #Add classes to class matrix
    malwareClass = classDictionary[fileStub]
    base2 = base[:]
    base2[int(malwareClass)-1] = 1
    labels.append(numpy.asarray(base2[:]))

#data_nGram = tools.normalize(numpy.asarray(data_nGram))

print("-- Built data set --")

for crossVal in range(numCrossVals):
    print("CrossVal: " + str(crossVal) + " | " + str(numCrossVals))
    trainSet, testSet = tools.createCrossValSets(crossVal, numCrossVals, len(data_cnn))

    cv_data_cnn = []
    cv_data_nGram = []
    cv_labels = []
    base = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    for num in trainSet:
        #Add values to data matrices (one matrix per classifier)
        cv_data_nGram.append(data_nGram[num-1])
        cv_data_cnn.append(data_cnn[num-1])

        #Add classes to class matrix
        cv_labels.append(labels[num-1])

    test_data_CNN = []
    test_data_nGram = []
    testLabels = []
    for num in testSet:
        # Add values to data matrices (one matrix per classifier)
        test_data_nGram.append(data_nGram[num-1])
        test_data_CNN.append(data_cnn[num-1])

        # Add test classes to class matrix
        testLabels.append(labels[num-1])


    #Run the training/testing

    #build CNN
    print("                         | Training -> CNN")
    CNN = TFConvNetwork.TFConvNetwork(X_DIMENSION, Y_DIMENSION, 9, cv_data_cnn, cv_labels, test_data_CNN, testLabels)
    CNN_DATA = CNN.trainAndClassify(CNN_ITERATIONS)
    print("                         | " + str(CNN_DATA[1]))
    CNN_accuracy += CNN_DATA[1]

    # build FFNN
    print("        | Training -> FFNN")
    FFNN = FeedForwardNeuralNetwork.FFNN(cv_data_nGram, cv_labels, test_data_nGram, testLabels, len(info["featureBatch"]), 9, iterations)
    FFNN_DATA = FFNN.trainAndClassify()
    print("          | " + str(FFNN_DATA[1]))
    NGRAM_accuracy += FFNN_DATA[1]

    print(str(crossVal).rjust(24) + " | Final Cross Val Result -> ")
    print("                         | CNN -> " + str(CNN_DATA[1]))
    print("                         | FFNN -> " + str(FFNN_DATA[1]))
    combVals = (CNN_DATA[1]*CNN_DATA[0]) + (FFNN_DATA[1]*FFNN_DATA[0])
    tempCombAcc = np.mean(np.argmax(testLabels, axis=1) == np.mean(np.argmax(combVals, axis=1)))
    combAccPercent += tempCombAcc
    print("                         | Combined -> " + str(combAccPercent))


finalAccuracyCNN = CNN_accuracy/numCrossVals
print("Final Cross Val accuracy --> CNN: " + str(finalAccuracyCNN))

finalAccuracyNGRAM = NGRAM_accuracy/numCrossVals
print("Final Cross Val accuracy --> FFNN: " + str(finalAccuracyNGRAM))

finalAccuracyComb = combAccPercent/numCrossVals
print("Final Cross Val accuracy --> Combined: " + str(finalAccuracyComb))

#!!!!!!!!!!!!!!!!!!!!!!
#Final Cross Val accuracy --> FFNN: 0.9502680109648564
#!!!!!!!!!!!!!!!!!!!!!!

#!!!!!!!!!!!!!!!!!!!!!!
#Final Cross Val accuracy --> CNN: 0.864428036765
#!!!!!!!!!!!!!!!!!!!!!!