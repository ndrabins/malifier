import sys
import distributedIGAlgorithm


def rebuildString(tasks, startNum):
    newString = ""
    for taskNum in range(startNum, len(tasks)):
        newString += tasks[taskNum] + "\n"

    return newString

file = open("/media/napster/data/train/TaskLog.txt", 'r')
tasks = file.readlines()
file.close()

endList = []

batch = []
start = 2100
for taskNum in range(start, len(tasks)):
    task = list(eval(tasks[taskNum]))
    batch.append(task)
    if taskNum % 100 == 0 and taskNum != start:
        new = distributedIGAlgorithm.runProcessBacklogItem(batch, taskNum-99)
        batch = []
        print(str(taskNum))



