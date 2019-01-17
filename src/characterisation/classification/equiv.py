import numpy as np

from pprint import pprint

from .sklearnHelper import filteredMap

"""
Jump point equivalence class method.
Checks if a userID (class) is within a subset of classes 
based on the biggest difference of probabilities between classes
"""
def jumpy(classPreds, targetIndeces):
    claccuracy = 0
    indAccuracy = 0
    filteredIDs = list(filteredMap().keys())

    # Loop through all of the predictions and their corresponding actual values
    # Then determine their accuracies.
    for i in np.arange(classPreds.shape[0]):
        # The target class and the array of predicted probabilities
        actualClass = targetIndeces[i]
        predictedProbs = classPreds[i]

        # Mapping each userID to their corresponding probabilities
        userMap = {userID: prob for userID, prob in zip(filteredIDs, predictedProbs)}
        predictedClass = keyFromValue(userMap, np.amax(predictedProbs))
        
        # Don't know if I'll run into issues using these indices later...
        # Yep, there are problems with this line only if a user class is not filtered out
        # And does not meet the supplied threshold
        actualIndex = filteredIDs.index(actualClass)
        # Index of the predicted class
        predictIndex = filteredIDs.index(predictedClass)

        sortedProbs = np.sort(predictedProbs)[::-1]
        diffs = np.absolute(np.diff(sortedProbs))
        
        # This isn't future proof, even though it's unlikely, multiple differences may have the
        # same value, so this could return either an array or a scalar
        # Index of the biggest difference in probabilities
        diffIndex = int(np.argmax(diffs))
        diffIndex = 1 if diffIndex == 0 else diffIndex

        # Splits the probabilities into 2 groups, one for the predicted equivalence class
        # Just dumping the rest in another one (which we don't care about)
        equivClass = sortedProbs[:int(diffIndex)]
        
        # Whether the target class appears in the equivalence class
        classPredicted = userMap[actualClass] in equivClass
        predicted = actualClass == predictedClass

        # Counting all the misses to later calculate the class accuracy
        if classPredicted:
            claccuracy += 1
        if predicted:
            indAccuracy += 1

    claccuracy = (claccuracy / classPreds.shape[0]) * 100
    indAccuracy = (indAccuracy / classPreds.shape[0]) * 100

    print("Jump points")
    print(f"Class Accuracy: {claccuracy:.2f}%")
    print(f"Individual Accuracy: {indAccuracy:.2f}%\n")


"""
Threshold function for users who have a score within the 95th percentile
Where score relates to the probability of the user being of the desired class.
"""
def scoreDistri(classPreds, targetIndeces, percentile=90):
    claccuracy = 0
    indAccuracy = 0

    for i in np.arange(classPreds.shape[0]):
        actualClass = targetIndeces[i]
        predictedProbs = classPreds[i]

        userMap = makeUserMap(predictedProbs)

        sortedProbs = np.sort(predictedProbs)[::-1]
        thresholdProb = sortedProbs[0] * (percentile / 100)

        sortedUsers = {userID: prob for userID, prob in userMap.items() if prob > thresholdProb}
        sortedUsers = sorted(sortedUsers.items(), key=lambda kv: kv[1], reverse=True)

        # Forming the array of equivalent classes
        equivClass = [users[0] for users in sortedUsers]
        predictedClass = equivClass[0]

        # Whether the target class appears in the equivalence class
        classPredicted = actualClass in equivClass
        predicted = actualClass == predictedClass        

        if classPredicted:
            claccuracy += 1
        if predicted:
            indAccuracy += 1

    claccuracy = (claccuracy / classPreds.shape[0]) * 100
    indAccuracy = (indAccuracy / classPreds.shape[0]) * 100

    print("Score Percentiles")
    print(f"Class Accuracy: {claccuracy:.2f}%")
    print(f"Individual Accuracy: {indAccuracy:.2f}%\n")

"""
Thresholded against users (not scores) within the 90th percentile
i.e. the 90th percentile of users when sorted by their scores
"""
def userCentiles(classPreds, targetIndeces, percentile=90, verbose=False):
    claccuracy = 0
    indAccuracy = 0

    filteredIDs = list(filteredMap().keys())
    numUsers = int(round(len(filteredIDs) * (1 / (100 - percentile))))

    for i in np.arange(classPreds.shape[0]):
        # The target class and the array of predicted probabilities
        actualClass = targetIndeces[i]
        predictedProbs = classPreds[i]

        # Mapping each userID to their corresponding probabilities
        userMap = makeUserMap(predictedProbs)
        
        # Sorting the users in descending order of possibilities
        sortedUsers = sorted(userMap.items(), key=lambda kv: kv[1], reverse=True)
        # Forming the equivalence class of users
        equivUsers = [users[0] for users in sortedUsers[:numUsers]]
        predictedClass = equivUsers[0]


        # Whether the target class appears in the equivalence class
        classPredicted = actualClass in equivUsers
        predicted = actualClass == predictedClass

        if verbose:
            print(f"Actual User {actualClass}")
            print(f"Predicted user {predictedClass}")
            print(f"Sorted Users {sortedUsers[:numUsers]}")
            print(f"Equivalent Users {equivUsers}")
            print(f"Correct Class Predicted {classPredicted}")
            print(f"Class User Predicted {predicted}\n\n")

        # Counting all the misses to later calculate the class accuracy
        if classPredicted:
            claccuracy += 1
        if predicted:
            indAccuracy += 1
    
    claccuracy = (claccuracy / classPreds.shape[0]) * 100
    indAccuracy = (indAccuracy / classPreds.shape[0]) * 100

    print("User Percentiles")
    print(f"Class Accuracy: {claccuracy:.2f}%")
    print(f"Individual Accuracy: {indAccuracy:.2f}%\n")

def keyFromValue(myDict, searchValue):
    return list(myDict.keys())[list(myDict.values()).index(searchValue)]

"""
Maps each userID to their corresponding probabilities
"""
def makeUserMap(predictedProbs):
    filteredIDs = list(filteredMap().keys())
    return {userID: prob for userID, prob in zip(filteredIDs, predictedProbs)}