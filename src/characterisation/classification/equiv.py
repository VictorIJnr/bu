import numpy as np

from pprint import pprint

from .sklearnHelper import filteredMap

def jumpy(classPreds, targetIndeces):
    claccuracy = 0
    indAccuracy = 0
    filteredIDs = list(filteredMap().keys())

    print(f"All of the filtered IDs {filteredIDs}")

    #Loop through all of the predictions and their corresponding actual values
    #Then determine their accuracies.
    for i in np.arange(classPreds.shape[0]):
        #The target class and the array of predicted probabilities
        actualClass = targetIndeces[i]
        predictedProbs = classPreds[i]

        #Mapping each userID to their corresponding probabilities
        userMap = {userID: prob for userID, prob in zip(filteredIDs, predictedProbs)}
        
        #Don't know if I'll run into issues using these indices later...
        #Yep there's somehow problems with this but I don't know exactly what's wrong
        actualIndex = filteredIDs.index(actualClass)
        #Index of the predicted class
        predictIndex = filteredIDs.index(keyFromValue(userMap, np.amax(predictedProbs)))

        print(f"Actual index {actualIndex}")
        print(f"Predicted index {predictIndex}\n\n")
        
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
        classPredicted = userMap[actualClass] in equivClass
        predicted = actualIndex == predictIndex

        #Counting all the misses to later calculate the class accuracy
        if not classPredicted:
            claccuracy += 1
        if not predicted:
            indAccuracy += 1

    claccuracy = (1 - (claccuracy / classPreds.shape[0])) * 100
    indAccuracy = (1 - (indAccuracy / classPreds.shape[0])) * 100

    print(f"Equivalence Class Array: {equivClass}")

    print(f"Class Accuracy: {claccuracy:.2f}%")
    print(f"Individual Accuracy: {indAccuracy:.2f}%")

"""
Threshold function for users who have a score within the 90th percentile
Where score relates to the probability of the user being of the desired class.
"""
def scoreDistri():
    pass

"""
Thresholded against users (not scores) within the 90th percentile
i.e. the 90th percentile of users when sorted by their scores
"""
def userCentiles(classPreds, targetIndeces):
    claccuracy = 0
    indAccuracy = 0
    filteredIDs = list(filteredMap().keys())

    for i in np.arange(classPreds.shape[0]):
        #The target class and the array of predicted probabilities
        actualClass = targetIndeces[i]
        predictedProbs = classPreds[i]

        #Mapping each userID to their corresponding probabilities
        userMap = {userID: prob for userID, prob in zip(filteredIDs, predictedProbs)}
        
        #Don't know if I'll run into issues using these indices later...
        #Yep there's somehow problems with this but I don't know exactly what's wrong
        actualIndex = filteredIDs.index(actualClass)
        #Index of the predicted class
        predictIndex = filteredIDs.index(keyFromValue(userMap, np.amax(predictedProbs)))

        print(f"Actual index {actualIndex}")
        print(f"Predicted index {predictIndex}\n\n")
        
        predictedProbs = np.sort(predictedProbs)[::-1]

def loopyHelpy(iterNum):
    #The target class and the array of predicted probabilities
    actualClass = targetIndeces[iterNum]
    predictedProbs = classPreds[iterNum]

    #Mapping each userID to their corresponding probabilities
    userMap = {userID: prob for userID, prob in zip(filteredIDs, predictedProbs)}
    
    #Don't know if I'll run into issues using these indices later...
    actualIndex = filteredIDs.index(actualClass)
    #Index of the predicted class
    predictIndex = filteredIDs.index(keyFromValue(userMap, np.amax(predictedProbs)))

    print(f"Actual index {actualIndex}")
    print(f"Predicted index {predictIndex}\n\n")
    
    predictedProbs = np.sort(predictedProbs)[::-1]

def keyFromValue(myDict, searchValue):
    return list(myDict.keys())[list(myDict.values()).index(searchValue)]