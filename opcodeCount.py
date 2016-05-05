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
    myFile = open(fName, 'r', encoding="ISO-8859-1")
    file = myFile.readlines()
    #Create binary search tree inbetween first and last values (for jumping)

    line = 0
    tokenDict = {}
    while not line >= len(file): #FIX THIS
        tokens = file[line].split()
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
        tokens = file[line].split()
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
                            tokensInfinite = file[newLine].split()
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