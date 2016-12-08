import numpy as np
import random

#=================================================
#=================================================
#   Any tools that are needed across multiple
#   files
#=================================================
#=================================================



#=================================================
#   The need to create an even set with
#   equal amounts in each class is used
#   by multiple elements such as the classifier
#   and feature selection functionality
#=================================================
def createEvenTestSet(classDictionary, ASM_FILES):
    classesSet = []

    temp = ASM_FILES[:]
    random.shuffle(temp)

    classVals = [1,2,3,4,5,6,7,8,9]
    min = len(ASM_FILES)
    counterHolder = []
    for counter in range(len(classVals)):
        counterHolder.append(0)
        num = list(classDictionary.values()).count(str(classVals[counter]))
        if num < min:
            min = num

    for item in range(len(temp)):
        if counterHolder[classVals.index(int(classDictionary[temp[item].split('/')[-1][:-4]]))] < min:
            classesSet.append(temp[item].split('/')[-1][:-4])
            counterHolder[classVals.index(int(classDictionary[temp[item].split('/')[-1][:-4]]))] += 1

    random.shuffle(classesSet)
    return classesSet


def buildAllSet(ASM_FILES):
    classesSet = []
    for item in range(len(ASM_FILES)):
        classesSet.append(ASM_FILES[item].split('/')[-1][:-4])

    random.shuffle(classesSet)

    return classesSet

#=================================================
#       Create cross-validation-set
#       grabs the train and test set
#       based off of the number of cross
#       vals as well as which item it is
#=================================================
def createCrossValSets(crossVal, numVals, mySetSize):
    temp = []
    for item in range(mySetSize):
        temp.append(item)

    part = int((mySetSize / numVals) * crossVal)
    part2 = int((mySetSize / numVals) * (crossVal + 1))
    testSet = temp[part:part2]

    sect1 = False
    if part > 0:
        section1 = temp[0:part]
        sect1 = True
    sect2 = False
    if part2 < (len(temp) - 1):
        section2 = temp[part2:len(temp)]
        sect2 = True

    if sect1 and sect2:
        trainSet = np.concatenate([section1, section2])
    elif sect1:
        trainSet = section1
    else:
        trainSet = section2

    return trainSet, testSet

#=================================================
#       Take a feature matrix with columns
#       for each feature and normalize all
#       values within
#=================================================
def normalize(data_set):
    count = 0
    print("Total Features = " + str(len(data_set[0])))
    print("Total Files = " + str(len(data_set)))
    for col in range(len(data_set[0])):
        temp = data_set[:, col]
        minVal = min(temp)
        maxVal = max(temp)

        if maxVal == 0:
            maxVal = 1
            #count += 1
            #print(str(count))

        for row in range(len(data_set)):
            data_set[row][col] = (data_set[row][col]-minVal)/(maxVal - minVal)

    return data_set
