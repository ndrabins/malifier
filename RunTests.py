import NeuralNetwork
import glob
import classDict
import getFeatures

TRAIN_FILE_PATH = "F:/train/"
MALWARE_FILE_PATH = TRAIN_FILE_PATH + "train/"

ASM_Files = glob.glob(MALWARE_FILE_PATH + "*.asm")
CLASS_Files = TRAIN_FILE_PATH + "trainLabels.txt"

INPUT_COUNT = 458
HIDDEN_NODES = 2
OUTPUT_COUNT= 1

#create neural network
MalwareNetwork = NeuralNetwork.NN(INPUT_COUNT, HIDDEN_NODES, OUTPUT_COUNT)

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
for file in trainSet:
    row = []
    featureList = getFeatures.getFeatures(file)
    row.append(featureList)
    fileStub = file.split('\\')[-1][:-4] #grab all but last chars in file name (removes .asm)
    malwareClass = classDictionary[fileStub]
    row.append([int(malwareClass)])

    toTrain.append(row)

#Run the training
MalwareNetwork.train(toTrain)

#assemble features in test set
toTest = []
for file in testSet:
    row = []
    featureList = getFeatures.getFeatures(file)
    row.append(featureList)

    fileStub = file.split('\\')[-1][:-4] #grab all but last chars in file name (removes .asm)
    malwareClass = classDictionary[fileStub]
    row.append([int(malwareClass)])

    toTest.append(row)

MalwareNetwork.test(toTest)