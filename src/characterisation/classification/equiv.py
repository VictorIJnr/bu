import numpy as np

from pprint import pprint

from .sklearnHelper import filteredMap

def jumpy(classPreds, targetIndeces):
    accuracy = 0
    filteredIDs = list(filteredMap().keys())

    #Loop through all of the predictions and their corresponding actual values
    #Then determine their accuracies.
    for i in np.arange(classPreds.shape[0]):
        #The target class and the array of predicted probabilities
        actualClass = targetIndeces[i]
        predictedProbs = classPreds[i]

        #Mapping each userID to their corresponding probabilities
        userMap = {userID: prob for userID, prob in zip(filteredIDs, predictedProbs)}
        
        #Don't know if I'll run into issues using these indices later...
        actualIndex = filteredIDs.index(actualClass)
        predictIndex = filteredIDs.index(keyFromValue(userMap, np.amax(predictedProbs)))
        
        predictedProbs = np.sort(predictedProbs)[::-1]
        diffs = np.diff(predictedProbs)
        
        #This isn't future proof, even though it's unlikely, multiple differences may have the
        #same value, so this could return either an array or a scalar
        #Index of the biggest difference in probabilities
        diffIndex = np.argmax(diffs)
        
        #Splits the probabilities into 2 groups, one for the predicted equivalence class
        #Just dumping the rest in another one (which we don't care about)
        equivClass = predictedProbs[:int(diffIndex)]
        
        #Whether the target class appears in the equivalence class
        predicted = userMap[actualClass] in equivClass

        if not predicted:
            accuracy += 1
    accuracy = (1 - (accuracy / classPreds.shape[0])) * 100

    print(f"Accuracy: {accuracy:.2f}%")

def keyFromValue(myDict, searchValue):
    return list(myDict.keys())[list(myDict.values()).index(searchValue)]