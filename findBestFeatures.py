import glob
import opcodeCount
import segmentCount

#############################################################################
#############################################################################
#        The purpose of this file is to gather some information about       #
#        the .asm files. We will gather (in a sense) features from the      #
#        .asm files to find out which features are most prevalent from      #
#        the set. Those which are most prevalent will be use true features  #
#        for the network later on.as                                        #
#############################################################################
#############################################################################


MALWARE_FILE_PATH = "/media/napster/data/train/train/"
SAVE_FILE_PATH = "/media/napster/data/train/"
ASM_Files = glob.glob(MALWARE_FILE_PATH + "*.asm")
fileFeatureList = []


#############################################################################
#############################################################################
#                           ANALYZE SEGMENT COUNT                           #
#      First get the segment count of all possible segments and store in    #
#      segFeatures.txt. Should only have to do this once!                   #
#############################################################################
#############################################################################

'''
segList = {}
segCounter = 0

segmentData = open("segFeatures.txt", 'w')

myFeatureList = []
for n in range(600):
    myFeatureList.append(0) #600 to be safe... should not hit this many. In the seg count paper they had 448
for file in ASM_Files:
    SEG_COUNTS = segmentCount.seg(file)
    for key in SEG_COUNTS:
        if key in segList:
            myFeatureList[segList[key]] = SEG_COUNTS[key]
        else:
            segList[key] = segCounter
            myFeatureList[segCounter] = SEG_COUNTS[key]
            segCounter += 1

    segmentData.write(str(fileFeatureList))

segmentData.close()
'''
#############################################################################
#############################################################################
#     Now all segment counts are in a file with a dictionary of their       #
#     address in the first line. I will now use this dictionary to          #
#     create a feature list                                                 #
#############################################################################
#############################################################################



#############################################################################
#############################################################################
#############################################################################
#############################################################################

#############################################################################
#############################################################################
# For this section I grabbed a list of all the 1grams in each file that occur
# more than 200 times. I base this method off of the kaggle paper
#############################################################################
#############################################################################
#matchedThresh = []
#for file in ASM_Files:                  #For each file
#    myDict = opcodeCount.op(file[:-4])       #Gather the single opcode count

#    for key in myDict:
#        if (myDict[key] >= 200) and (not (key in matchedThresh)):
#            matchedThresh.append(key)

#file = open(SAVE_FILE_PATH+"opcode.txt", 'a')
#file.write(str(matchedThresh))
#file.close()

#############################################################################
#############################################################################
# Now I will be using my list, after preprocessing (in preprocessing.py) to
# create a new list of ngrams.
#############################################################################
#############################################################################

file = open("/media/napster/data/train/final_one_grams.txt", 'r')
oneGrams = list(eval(file.readline()))
file.close()
opcodeCount.nGramsTwoThroughFour(ASM_Files, oneGrams)