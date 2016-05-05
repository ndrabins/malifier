import NeuralNetwork
import glob
import classDict
import getFeatures

MALWARE_FILE_PATH = "D:/train/"
TRAIN_FILE_PATH = MALWARE_FILE_PATH + "train/"

ASM_Files = glob.glob(MALWARE_FILE_PATH + "*.asm")
CLASS_Files = MALWARE_FILE_PATH + "trainLabels"

INPUT_COUNT = 0
HIDDEN_LAYERS = 2
OUTPUT_COUNT= 9

#create neural network
MalwareNetwork = NeuralNetwork.NN(INPUT_COUNT, HIDDEN_LAYERS, OUTPUT_COUNT)

#Get the lookup table for Malware Classes
classDictionary = classDict.getMalClasses(CLASS_Files)

#Create Simple Train/Test partition
part = (len(ASM_Files)/3)*2
trainSet = ASM_Files[:part]
testSet = ASM_Files[part:]

#Assemble features in train set
toTrain = []
for file in trainSet:
    row = []
    featureList = getFeatures.getFeatures(file)
    row.append(featureList)

    fileStub = file.split('/')[-1][:-4] #grab all but last chars in file name (removes .asm)
    malwareClass = classDictionary[fileStub]
    row.append([malwareClass])

    toTrain.append(row)

#Run the training
MalwareNetwork.train(toTrain)

#assemble features in test set
toTest = []
for file in testSet:
    row = []
    featureList = getFeatures.getFeatures(file)
    row.append(featureList)

    fileStub = file.split('/')[-1][:-4] #grab all but last chars in file name (removes .asm)
    malwareClass = classDictionary[fileStub]
    row.append([malwareClass])

    toTest.append(row)


MalwareNetwork.test(toTest)
