import NeuralNetwork
import glob

import numpy as np
import tools
import TFConvNetwork
import classDict
import getFeatures
import numpy
import distributedIGAlgorithm

TRAIN_FILE_PATH = "/media/napster/data/train/" #"F:/train/"
MALWARE_FILE_PATH = TRAIN_FILE_PATH + "train/"

ASM_Files = glob.glob(MALWARE_FILE_PATH + "*.asm")
CLASS_Files = TRAIN_FILE_PATH + "trainLabels.txt"

INPUT_COUNT = 458
HIDDEN_NODES = 2
OUTPUT_COUNT = 9

#create neural network
MalwareNetwork = NeuralNetwork.NN(INPUT_COUNT, HIDDEN_NODES, OUTPUT_COUNT)

#build CNN
CNN_ITERATIONS = 200000
X_DIMENSION = 16
Y_DIMENSION = 64
CNN = TFConvNetwork.TFConvNetwork(X_DIMENSION, Y_DIMENSION)

#build info dict for getting CNN features
info = {}
info["X"] = X_DIMENSION
info["Y"] = Y_DIMENSION

#Get the lookup table for Malware Classes
classDictionary = classDict.getMalClasses(CLASS_Files)

#Create Simple Train/Test partition
mySet = tools.createEvenTestSet(classDictionary, ASM_Files)


#Assemble features in train set
#==================================
# This is the section that will take the longest time running since it will actually be extracting the features of
# the malware for each file. This could be a lengthy process.
#==================================
accuracy = 0

for crossVal in range(10):

    part = int((len(mySet) / 10) * crossVal)  # Creates a partition at 2/3
    part2 = int((len(mySet) / 10) * (crossVal+1))  # Creates a partition at 2/3
    testSet = mySet[part:part2]  # ASM_Files[part:]  #Second 1/3 is testing

    sect1 = False
    if part > 0:
        section1 = mySet[0:part]
        sect1 = True
    sect2 = False
    if part2 < (len(mySet) - 1):
        section2 = mySet[part2:len(mySet)]
        sect2 = True

    if sect1 and sect2:
        trainSet = np.concatenate([section1, section2])
    elif sect1:
        trainSet = section1
    else:
        trainSet = section2

    toTrain = []
    data = []
    labels = []
    base = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    for fileStub in trainSet:
        row = []
        #fileStub = file.split('/')[-1][:-4]  # grab all but last chars in file name (removes .asm)

        #featureListNGRAM = getFeatures.getFeatures(MALWARE_FILE_PATH + fileStub, "NGRAM")  # file

        featureList = getFeatures.getFeatures(MALWARE_FILE_PATH + fileStub, "CNN", info) #file

        #check for error
        if (len(featureList) == 0):
            continue

        data.append(numpy.asarray(featureList[0]))
        #row.append(featureList)
        malwareClass = classDictionary[fileStub]
        base2 = base[:]
        base2[int(malwareClass)-1] = 1
        #row.append(numpy.asarray(base2[:]))
        labels.append(numpy.asarray(base2[:]))

    toTrain.append(data)
    toTrain.append(labels)

    #Run the training
    #MalwareNetwork.train(toTrain)
    CNN.train(toTrain, CNN_ITERATIONS)

    #assemble features in test set
    toTest = []
    testData = []
    testLabels = []
    for fileStub in testSet:
        row = []
        #fileStub = file.split('/')[-1][:-4]  # grab all but last chars in file name (removes .asm)
        featureList = getFeatures.getFeatures(MALWARE_FILE_PATH + fileStub, "CNN")
        row.append(featureList)

        #check for error
        if (len(featureList) == 0):
            continue

        malwareClass = classDictionary[fileStub]
        row.append([int(malwareClass)])

        testData.append(numpy.asarray(featureList[0]))
        # row.append(featureList)
        malwareClass = classDictionary[fileStub]
        base2 = base[:]
        base2[int(malwareClass) - 1] = 1
        # row.append(numpy.asarray(base2[:]))
        testLabels.append(numpy.asarray(base2[:]))

    toTest.append(testData)
    toTest.append(testLabels)

    # Run the training
    # MalwareNetwork.train(toTrain)
    accuracy += CNN.test(toTest)

finalAccuracy = accuracy/10
print("Final Cross Val accuracy: " + str(finalAccuracy))
#MalwareNetwork.test(toTest)