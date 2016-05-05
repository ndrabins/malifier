#############################################################################
#############################################################################
#        Segment counts is a feature that has shown to be very effective    #
#        in classifying malware. It is quite straightforward - just         #
#        the headers before each file. Only a certain number of these       #
#        segment counts will be included as features, but those top few     #
#        will be chosen based on the total number from all files.           #
#        Therefore, this returns counts of all segments.                    #
#############################################################################
#############################################################################

def seg(fName):

    SEG_COUNTS = {}
    #Step 1 - grab the 1-gram features
    file = open(fName, 'r', encoding="ISO-8859-1")
    for line in file:
        tokens = line.split()
        segment = tokens[0].split(':')[0]
        if not segment in SEG_COUNTS:
            SEG_COUNTS[segment] = 1
        else:
            SEG_COUNTS[segment] += 1

    file.close()
    return SEG_COUNTS