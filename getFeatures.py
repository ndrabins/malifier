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

    # non functional

    #myDict = opcodeCount.op(file)
    #fileFeatureList.append(myDict)

    #########################################################



    #########################################################
    # FEATURE SET 3 -- Get Header Image ---------------------
    #########################################################

    #number of bytes -> 1600

    image = extract_header_image.extract_picture_from_bytes(file + ".bytes", 16*64)

    #all ?? is bytes file so will not use it to classify....
    #must handle empty list on return
    if (len(image) == 0):
        return []

    fileFeatureList.append(image)

    #########################################################


    return fileFeatureList
