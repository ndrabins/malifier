import NeuralNetwork
import glob

import TFConvNetwork
import classDict
import getFeatures
import numpy

TRAIN_FILE_PATH = "/media/napster/data/train/" #"F:/train/"
MALWARE_FILE_PATH = TRAIN_FILE_PATH + "train/"

ASM_Files = glob.glob(MALWARE_FILE_PATH + "*.asm")
CLASS_Files = TRAIN_FILE_PATH + "trainLabels.txt"

INPUT_COUNT = 458
HIDDEN_NODES = 2
OUTPUT_COUNT= 9

#create neural network
MalwareNetwork = NeuralNetwork.NN(INPUT_COUNT, HIDDEN_NODES, OUTPUT_COUNT)

#build CNN

CNN = TFConvNetwork.TFConvNetwork(16, 64)

#Get the lookup table for Malware Classes
classDictionary = classDict.getMalClasses(CLASS_Files)

#Create Simple Train/Test partition
part = int((len(ASM_Files)/3)*2) #Creates a partition at 2/3
trainSet = ASM_Files[:part] #The first 2/3 are training
testSet = ASM_Files[part:]#ASM_Files[part:]  #Second 1/3 is testing

#Assemble features in train set
#==================================
# This is the section that will take the longest time running since it will actually be extracting the features of
# the malware for each file. This could be a lengthy process.
#==================================
toTrain = []
data = []
labels = []
base = [0, 0, 0, 0, 0, 0, 0, 0, 0]

for file in trainSet:
    row = []
    fileStub = file.split('/')[-1][:-4]  # grab all but last chars in file name (removes .asm)

    featureListNGRAM = getFeatures.getFeatures(MALWARE_FILE_PATH + fileStub, "NGRAM")  # file

    featureList = getFeatures.getFeatures(MALWARE_FILE_PATH + fileStub, "CNN") #file

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
CNN.train(toTrain, 500000)

#assemble features in test set
toTest = []
testData = []
testLabels = []
for file in testSet:
    row = []
    fileStub = file.split('/')[-1][:-4]  # grab all but last chars in file name (removes .asm)
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
CNN.test(toTest)


#MalwareNetwork.test(toTest)