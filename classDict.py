def getMalClasses(fName):
    classDictionary = {}
    file = open(fName, 'r')
    for line in file:
        nameAndClass = line.split()
        classDictionary[nameAndClass[0]] = nameAndClass[1]

    return classDictionary