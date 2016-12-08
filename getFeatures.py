import extract_header_image
import opcodeCount
import segmentCount

    #############################################################################
    #############################################################################
    #        Get features will return the feature set for all of the files      #
    #        as a list of values. Each feature will be added in as it is        #
    #        implemented.                                                       #
    #############################################################################
    #############################################################################


TRAIN_FILE_PATH = "/media/napster/data/train/"

def getFeatures(file, feature, info={}):

    segList = {}
    segCounter = 0

    #########################################################
    # FEATURE SET  1 -- Get Seg Count -----------------------
    #########################################################

    #########################################################
    #     Grab the segCount dictionary that was created     #
    #     by findBestFeatures.                              #
    #########################################################

    #functional

    #SegmentFile = open('segFeatures.txt', 'r', encoding="ISO-8859-1")
    #SegmentDictionary = eval(SegmentFile.readline()) #grab dictionary from first line
    #SegmentFile.close()

    #Prep the feature list for segment counts
    #for feature in range(len(SegmentDictionary)):
    #    fileFeatureList.append(0)

    #Get Seg counts
    #SEG_COUNTS = segmentCount.seg(file)

    #Add Seg counts to feature list
    #for segment in SEG_COUNTS:
    #    if (segment in SEG_COUNTS and segment in SegmentDictionary):
    #        fileFeatureList[SegmentDictionary[segment]] = SEG_COUNTS[segment]
    #    elif (segment in SegmentDictionary and SegmentDictionary[segment] in fileFeatureList):
    #        fileFeatureList[SegmentDictionary[segment]] = 0
    #########################################################




    #########################################################
    # FEATURE SET 2 -- Get N-gram Counts --------------------
    #########################################################
    if (feature == "NGRAM"):
        # non functional
        try:
            file = open(TRAIN_FILE_PATH + "nGramFeatures/" + file + ".txt", 'r')
            finderNGramDict = dict(eval(file.readline()))
            file.close()
        except FileNotFoundError:
            # for some reason there are files missing
            return []

        myBatchMatrix = []
        for item in range(len(info["featureBatch"])):
            if info["featureBatch"][item] in finderNGramDict:
                myBatchMatrix.append(finderNGramDict[info["featureBatch"][item]])
            else:
                myBatchMatrix.append(0)


        #myDict = opcodeCount.op(file)
        #preprocessing.trim1GramOpcodeDicts(myDict)
        return myBatchMatrix

    #########################################################



    #########################################################
    # FEATURE SET 3 -- Get Header Image ---------------------
    #########################################################
    if (feature == "CNN"):

        #number of bytes -> 1600

        image = extract_header_image.extract_picture_from_bytes(file + ".bytes", info["X"]*info["Y"])

        #all ?? is bytes file so will not use it to classify....
        #must handle empty list on return
        if (len(image) == 0):
            return []

        return image

    #########################################################


