import re

#############################################################################
#############################################################################
#        This file is central to a certain feature of the Malware: n-grams  #
#        To gather our n-grams we are using the methodology used by the     #
#        First place winners of the contest on the dataset that we are      #
#        using. The winners are Xiaozhou Wang, Jiwei Liu, and Xueer Chen.   #
#        To do this, first we find all notable "tokens" or and space        #
#        separated text in the file. Importantly, we follow any jumps in    #
#        file that are unconditional, counting any operations that are      #
#        in an infinite loop as automatically notable. Returned from this   #
#        functions is just a list of tokens and their counts for each file. #
#############################################################################
#############################################################################

def op(fName):
    LOOP_THRESHOLD = 200

    #Step 1 - grab the 1-gram features
    myFile = open(fName + ".asm", 'r', encoding="ISO-8859-1")
    file = myFile.readlines()
    #Create binary search tree inbetween first and last values (for jumping)

    line = 0
    tokenDict = {}
    while not line >= len(file): #FIX THIS
        #split the line into two halves, the address and byte data, then opcodes and others
        #take second half and split it to get tokens
        tokens = getImportants(file[line])
        for token in range(len(tokens)):
            if tokens[token] in tokenDict:
                tokenDict[tokens[token]] += 1
            else:
                tokenDict[tokens[token]] = 1
        line += 1


    line = 0
    jumpRows = []
    locRows = []
    while not line >= len(file):
        #split the line into two halves, the address and byte data, then opcodes and others
        #take second half and split it to get tokens
        tokens = getImportants(file[line])

        for token in range(len(tokens)):
            if tokens[token] == "jmp":
                if (tokens[-1][0:4] == "loc_"):
                    #have we already gotten to this jump?
                    if line in jumpRows:
                        infiniteTokens = {}
                        #If we are here we are in an infinite loop
                        newLine = locRow
                        while not newLine == line: #now we need to find all of the tokens until we hit the beginning of infinite loop again
                            inJump = False

                            tokensInfinite = getImportants(file[newLine])

                            for tokenInfinite in range(len(tokensInfinite)):
                                if tokensInfinite[tokenInfinite] == "jmp":
                                    if (tokensInfinite[-1][0:4] == "loc_"): #if we hit another jmp first then we know that we have already hit it before
                                        try:
                                            spot = tokensInfinite[-1].split('_')[1]
                                            if "+" in spot:
                                                spot = spot.split('+')[0]
                                            infiniteJmpLoc = int(spot, 16) #so we find its location
                                        except:
                                            print(fName + ": ")
                                            print("     " + str(newLine) + " - ")
                                            print("     " + tokensInfinite[-1].split('_')[1])

                                        jmpLoc = locRows[jumpRows.index(infiniteJmpLoc)]
                                        newLine = jmpLoc - 1 #and go to it
                                        break

                                #here we need to grab the token so that we can multiply it by 10
                                if tokensInfinite[tokenInfinite] in tokenDict:
                                    tokenDict[tokensInfinite[tokenInfinite]] += 1
                                else:
                                    tokenDict[tokensInfinite[tokenInfinite]] = 1

                            newLine += 1

                        #add infinite tokens dictionary to regular token (multiplied by 10)
                        for key in infiniteTokens:
                            tokenDict[key] += infiniteTokens[key] + LOOP_THRESHOLD

                        line = max(jumpRows)

                    else:
                        jumpRows.append(line)

                        try:
                            spot = tokens[-1].split('_')[1]
                            if "+" in spot:
                                spot = spot.split('+')[0]
                            loc = int(spot, 16) #splits loc_402394 (byte addres) to get 402394
                        except:
                            print(fName + ": ")
                            print("     " + str(line) + " - ")
                            print("     " + tokens[-1].split('_')[1])

                        #WE NEED TO FIND THE LOCATION TO JUMP TO
                        #Begin iteration through search tree
                        locFound = False
                        counter = 0 #just to be safe
                        start = 0
                        end = len(file)-1
                        while not locFound:
                            mid = round((end-start)/2)+start
                            data = int(file[mid].split()[0].split(':')[1], 16)
                            if data > loc:
                                end = mid - 1
                            elif data < loc:
                                start = mid
                            else:
                                locFound = True
                                locRow = mid

                            counter +=1
                            if counter > 10000:
                                #if here then the location must not exist (happens in at least one file
                                #
                                locRow = line + 1
                                #
                                break

                        locRows.append(locRow)
                        line = locRow - 1
                        break

        line += 1
    myFile.close()

    return tokenDict


#this function is a monster....while I have the file in memory to find the ngrams at all I am going to go ahead and try and collect
#the feature value...aka, how many times that ngram appears in that file.
def nGramsTwoThroughFour(fNames, starter):
    # Step 2 - grab the 2-gram 3-gram and 4-gram features
    #nGramsList = [[], [], []]
    #a total list of all nGrams for allFiles. This contains a list of the unique files that have the nGram
    #nGramsDict = {}

    #214 was where it died last time
    for fName in range(661, len(fNames)):
        if fName % 20 == 0:
            print(str(fName) + "  |  " + str(len(fNames)))

        newDict = {}
        myFile = open(fNames[fName], 'r', encoding="ISO-8859-1")
        file = myFile.read()

        ngrams = {}

        for start in starter:
            line = 0
            tokenDict = {}
            reg = re.compile('(' + start + '(?:[\x20\t]+[\S]+){1,3})')
            m = reg.findall(file)

            for match in m:
                group = match.split()
                for num in range(2, 4):
                    if len(group) >= num:
                        myGram = ' '.join(group[0:num])

                        #if myGram in nGramsList[len(myGram.split())-2]:
                            # if this is the first time that this ngram has been found in this file then add it to the list of
                            # unique files that have this ngram
                        #   if not fName in nGramsDict[myGram]:
                               #add filenumber, does not really matter what it is as long as it shows that it is unique.
                               #at the end the length of the list will show how many files the ngram is in
                        #       nGramsDict[myGram].append(fName)

                               #if we reach here it is the first time that this nGram has been found in this file
                        if myGram in newDict:
                        #   else:
                            newDict[myGram] += 1
                        else:

                        #else:
                        #    nGramsList[len(myGram.split()) - 2].append(myGram)
                        #    nGramsDict[myGram] = [fName]

                            # if we reach here it is the first time that this nGram has been found in this file
                            newDict[myGram] = 1

        fileStub = fNames[fName].split('/')[-1][:-4]
        myStorage = open("/media/napster/data/train/nGramFeatures/" + fileStub + ".txt", 'w')
        newDict = {k: v for k, v in newDict.items() if v > 1}
        myStorage.write(str(newDict))
        myStorage.close()

def getImportants(line):
    tokens = line.split('\t')
    if len(tokens) > 2:
        tokens = ' '.join(tokens[3:])
        return tokens.split()
    return []


def howManyOccurenceOfOneGram(fNames, ones):
    onesCount = []
    for one in range(len(ones)):
        onesCount.append(0)

    for fName in fNames:
        myFile = open(fName, 'r', encoding="ISO-8859-1")
        file = myFile.read()

        for one in range(len(ones)):
            if file.find(ones[one]) > -1:
                onesCount[one] += 1
        myFile.close()

    return onesCount


#nGramsTwoThroughFour(['/media/napster/data/train/train/0A32eTdBKayjCWhZqDOQ'], ['mov'])