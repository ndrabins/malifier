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

    classVals = [1,2,3,4,5,6,7,8,9]
    min = len(ASM_FILES)
    counterHolder = []
    for counter in range(len(classVals)):
        counterHolder.append(0)
        num = list(classDictionary.values()).count(str(classVals[counter]))
        if num < min:
            min = num

    for item in range(len(ASM_FILES)):
        if counterHolder[classVals.index(int(classDictionary[ASM_FILES[item].split('/')[-1][:-4]]))] < min:
            classesSet.append(ASM_FILES[item].split('/')[-1][:-4])
            counterHolder[classVals.index(int(classDictionary[ASM_FILES[item].split('/')[-1][:-4]]))] += 1

    return classesSet