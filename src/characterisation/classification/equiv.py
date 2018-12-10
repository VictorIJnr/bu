import numpy as np

from pprint import pprint

from .sklearnHelper import filteredMap

def jumpy(classPreds, targetIndeces):
    filteredIDs = list(filteredMap().keys())

    #Loop through all of the predictions and their corresponding actual values
    #Then determine their accuracies.
    for i in np.arange(classPreds.shape[0]):
        #The target class and the array of predicted probabilities
        actualClass = targetIndeces[i]
        predictedProbs = classPreds[i]

        #Mapping each userID to their corresponding probabilities
        userMap = {userID: prob for userID, prob in zip(filteredIDs, predictedProbs)}
        
        #

    #Those 0s at the end will change to i in a loop
    firstClass = targetIndeces[0]
    firstProbs = classPreds[0]
    
    userMap = {userID: prob for userID, prob in zip(list(filteredMap().keys()), firstProbs)}
    userKeyList = list(userMap.keys())


    print(f"User key list {filteredIDs}")
    print(f"Key list type {type(filteredIDs)}")

    print(f"First Prob type {type(firstProbs)}")
    print(f"Unsorted Probabilities {firstProbs}")
    print(f"First Class {firstClass}")

    actualIndex = filteredIDs.index(firstClass)
    predictIndex = filteredIDs.index(keyFromValue(userMap, np.amax(firstProbs)))

    print(f"Predicted Index {predictIndex}")
    print(f"Actual Index {actualIndex}")

    firstProbs = np.sort(firstProbs)[::-1]

    diffs = np.diff(firstProbs)

    #This isn't future proof, even though it's unlikely, multiple differences may have the
    #same value, so this could return either an array or a scalar
    firstMax = np.argmax(diffs)

    print(firstProbs)
    print(diffs)

    print(diffs.shape)
    print(np.argmax(diffs))

    print("First Max")
    print(firstMax)
    print(type(firstMax))

    #Divides all the probabilities into 2, one predicted equivalence class and the rest
    equivClass = firstProbs[:int(firstMax)]
    pprint(equivClass)

    #Checks if the actual class was contained in the respective equivalence class
    predicted = userMap[firstClass] in equivClass
    print(predicted)

def keyFromValue(myDict, searchValue):
    return list(myDict.keys())[list(myDict.values()).index(searchValue)]