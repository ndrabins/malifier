import opcodeCount
import segmentCount

    #############################################################################
    #############################################################################
    #        Get features will return the feature set for all of the files      #
    #        as a list of values. Each feature will be added in as it is        #
    #        implemented.                                                       #
    #############################################################################
    #############################################################################


def getFeatures(file):

    segList = {}
    segCounter = 0
    fileFeatureList = []

    #########################################################
    # FEATURE SET  1 -- Get Seg Count -----------------------
    #########################################################

    #########################################################
    #     Grab the segCount dictionary that was created     #
    #     by findBestFeatures.                              #
    #########################################################
    SegmentFile = open('segFeatures.txt', 'r', encoding="ISO-8859-1")
    SegmentDictionary = eval(SegmentFile.readline()) #grab dictionary from first line
    SegmentFile.close()

    #Prep the feature list for segment counts
    for feature in range(len(SegmentDictionary)):
        fileFeatureList.append(0)

    #Get Seg counts
    SEG_COUNTS = segmentCount.seg(file)

    #Add Seg counts to feature list
    for segment in SEG_COUNTS:
        fileFeatureList[SegmentDictionary[segment]] = SEG_COUNTS[segment]
    #########################################################




    #########################################################
    # FEATURE SET 2 -- Get N-gram Counts --------------------
    #########################################################
    #myDict = opcodeCount.op(file)
    #fileFeatureList.append(myDict)

    #########################################################

    return fileFeatureList
