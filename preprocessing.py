import classDict
import string
import glob
import opcodeCount
import math

def trimOneGramsBasedOnCount():
    file = open("/media/napster/data/train/one_gram_count.txt", 'r')
    both = list(eval(file.readline()))
    grams = both[0]
    count = both[1]
    file.close()

    maxFiles = max(count)
    newCount = []

    for num in range(len(count)):
        if count[num] > ((8/9) * maxFiles):
            newCount.append(grams[num])
            #print(grams[num])

    #print(len(newCount))


    file = open("/media/napster/data/train/final_one_grams.txt", 'w')
    file.write(str(newCount))
    file.close()

trimOneGramsBasedOnCount()

def representsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def getOneGramsCount():
    file = open("/media/napster/data/train/one_grams.txt", 'r')
    myOnes = eval(file.readline())
    file.close()

    MALWARE_FILE_PATH = "/media/napster/data/train/train/"
    SAVE_FILE_PATH = "/media/napster/data/train/"
    ASM_Files = glob.glob(MALWARE_FILE_PATH + "*.asm")

    num = opcodeCount.howManyOccurenceOfOneGram(ASM_Files, myOnes)

    newFile = open("/media/napster/data/train/one_gram_count.txt", 'w')
    newFile.write(str(myOnes))
    newFile.write('/n')
    newFile.write(str(num))
    newFile.close()

def pareOneGrams():
    file = open("/media/napster/data/train/opcode.txt", 'r')

    instructs = open("/media/napster/data/train/instructions.txt", 'r')
    myList = eval(file.readline())
    myInstList = []
    registers = ["EAX", "EBX", "ECX", "EDX", "ESI", "EDI", "EBP", "EIP", "ESP"]

    inst = instructs.readline()
    while inst:
        myInstList.append(inst.split()[0])
        inst = instructs.readline()

    newlist = []
    for item in myList:
        if len(item) > 1 and not representsInt(item[0]) and not item[0] == "*" and not item[0:2] == "[e" and not item[0] == "." and not item.upper() in registers:
            if len(item) > 3:
                if (not item[0:3] == "sub") and (not item[0:3] == "loc"):
                    hasPunct = False
                    for char in item:
                        if char in string.punctuation:
                            hasPunct = True
                            break
                    if not hasPunct:
                        newlist.append(item)
            else:
                hasPunct = False
                for char in item:
                    if char in string.punctuation:
                        hasPunct = True
                        break
                if not hasPunct:
                    newlist.append(item)

    #instrFinal = []
    #nonInstFinal = []
    #for item in newlist:
    #    if not item.upper() in myInstList:
    #        nonInstFinal.append(item)
    #        print(item)
    #    else:
    #        instrFinal.append(item)

    file.close()
    instructs.close()

    file = open("/media/napster/data/train/one_grams.txt",'w')
    file.write(str(newlist))
    file.close()

    #54 out of 81 of the X86 instructions are in the list.
    #these will all be included
    #print(len(instrFinal))
    #print(len(myInstList))



