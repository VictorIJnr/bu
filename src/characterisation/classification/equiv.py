import numpy as np
import pandas as pd

from collections import defaultdict
from enum import Enum
from pprint import pprint

from .sklearnHelper import filteredMap

Equivs = Enum("Equivs", "JUMP SCORE_DIST PERCENTILES")

"""
Jump point equivalence class method.
Checks if a userID (class) is within a subset of classes 
based on the biggest difference of probabilities between classes

dataset - the dataset which predictions are tested on 
          currently a string but should be changed to an enum sometime
"""
def jumpy(dataset, classPreds, filteredIDs=None):
    filteredIDs = list(filteredMap(dataset=dataset).keys()) if filteredIDs is None else filteredIDs

    classProbs = {classID : probability for classID, probability in zip(filteredIDs, classPreds)}
    predictedClass = keyFromValue(classProbs, np.amax(classPreds))

    sortedProbs = np.sort(classPreds)[::-1]
    diffs = np.absolute(np.diff(sortedProbs))
    
    # This isn't future proof, even though it's unlikely, multiple differences may have the
    # same value, so this could return either an array or a scalar
    # Index of the biggest difference in probabilities
    diffIndex = int(np.argmax(diffs))
    diffIndex = 1 if diffIndex == 0 else diffIndex

    # Splits the probabilities into 2 groups, one for the predicted equivalence class
    # Just dumping the rest in another one (which we don't care about)
    # This won't scale for multiple classes with identical probabilities
    equivClass = [keyFromValue(classProbs, prob) for prob in sortedProbs[:int(diffIndex)]]

    return equivClass

"""
Score Distribution equivalence class method.
"""
def scoreDistri(dataset, classPreds, percentile=90):
    userMap = makeUserMap(dataset, classPreds)

    # Sorting scores to determine those within the given percentile
    sortedProbs = np.sort(classPreds)[::-1]
    thresholdProb = sortedProbs[0] * (percentile / 100)

    sortedUsers = {userID: prob for userID, prob in userMap.items() if prob > thresholdProb}
    sortedUsers = sorted(sortedUsers.items(), key=lambda kv: kv[1], reverse=True)

    # Forming the equivalence class array
    equivClass = [users[0] for users in sortedUsers]

    return equivClass

"""
User Percentiles equivalence class method.
"""
def userCentiles(dataset, classPreds, percentile=90, filteredIDs=None):
    filteredIDs = list(filteredMap(dataset=dataset).keys()) if filteredIDs is None else filteredIDs
    numUsers = int(round(len(filteredIDs) * (1 / (100 - percentile))))

    # Mapping each userID to their corresponding probabilities
    userMap = makeUserMap(dataset, classPreds)
    
    # Sorting the users in descending order of possibilities
    sortedUsers = sorted(userMap.items(), key=lambda kv: kv[1], reverse=True)
    # Forming the equivalence class of users
    equivClass = [users[0] for users in sortedUsers[:numUsers]]

    return equivClass

"""
Experimental method for running experiments. What else?

Using the jump-point equivalence class method; with a set of test data, determines both
the individual (user) accuracy and class accuracy of a model, upon a set of predictions
"""
def jumpyExperimental(classPreds, targetIndeces, dataset="worldbuilding", individual=False):
    claccuracy = 0
    indAccuracy = 0
    filteredIDs = list(filteredMap(dataset=dataset).keys())
    equivClasses = []

    classSizeCounter = defaultdict(lambda: 0)
    exactPredCounter = defaultdict(lambda: 0)
    
    experimentDF = []

    # Loop through all of the predictions and their corresponding actual values
    # Then determine their accuracies.
    for i in np.arange(classPreds.shape[0]):
        testInstance = defaultdict(lambda: None)

        # The target class and the array of predicted probabilities
        actualClass = targetIndeces[i]
        predictedProbs = classPreds[i]

        equivClass = jumpy(dataset, predictedProbs, filteredIDs=filteredIDs)
        
        # Whether the target class appears in the equivalence class
        classPredicted = actualClass in equivClass
        predicted = actualClass == equivClass[0]

        # Counting all the hits to later calculate the class accuracy
        if classPredicted:
            claccuracy += 1
        if predicted:
            indAccuracy += 1

        equivClasses.append(equivClass)
        
        testInstance["Actual Class"] = actualClass
        testInstance["Predicted Class"] = equivClass[0]
        testInstance["Equiv Class Size"] = len(equivClass)
        testInstance["Class Predicted"] = classPredicted
        testInstance["User Predicted"] = predicted

        experimentDF.append(testInstance)

    claccuracy = (claccuracy / classPreds.shape[0]) * 100
    indAccuracy = (indAccuracy / classPreds.shape[0]) * 100

    print("Jump points")
    print(f"Class Accuracy: {claccuracy:.2f}%")
    print(f"Individual Accuracy: {indAccuracy:.2f}%\n")

    experimentDF = pd.DataFrame(experimentDF)

    if individual:
        return experimentDF
    else:
        return claccuracy, indAccuracy, equivClasses

"""
Threshold function for users who have a score within the 90th percentile
Where score relates to the probability of the user being of the desired class.
"""
def scoreDistriExperimental(classPreds, targetIndeces, percentile=90, dataset="worldbuilding"):
    claccuracy = 0
    indAccuracy = 0

    equivClasses = []

    for i in np.arange(classPreds.shape[0]):
        actualClass = targetIndeces[i]
        predictedProbs = classPreds[i]

        # Forming the array of equivalent classes
        equivClass = scoreDistri(dataset, predictedProbs, percentile)
        predictedClass = equivClass[0]

        # Whether the target class appears in the equivalence class
        classPredicted = actualClass in equivClass
        predicted = actualClass == predictedClass        

        if classPredicted:
            claccuracy += 1
        if predicted:
            indAccuracy += 1

        equivClasses.append(equivClass)

    claccuracy = (claccuracy / classPreds.shape[0]) * 100
    indAccuracy = (indAccuracy / classPreds.shape[0]) * 100

    print("Score Percentiles")
    print(f"Class Accuracy: {claccuracy:.2f}%")
    print(f"Individual Accuracy: {indAccuracy:.2f}%\n")

    return claccuracy, indAccuracy, equivClasses

"""
Thresholded against users (not scores) within the 90th percentile
i.e. the 90th percentile of users when sorted by their scores
"""
def userCentilesExperimental(classPreds, targetIndeces, percentile=90, verbose=False, dataset="worldbuilding"):
    claccuracy = 0
    indAccuracy = 0

    filteredIDs = list(filteredMap(dataset=dataset).keys())
    equivClasses = []

    for i in np.arange(classPreds.shape[0]):
        # The target class and the array of predicted probabilities
        actualClass = targetIndeces[i]
        predictedProbs = classPreds[i]

        # Forming the equivalence class of users
        equivUsers = userCentiles(dataset, predictedProbs, percentile, filteredIDs=filteredIDs)
        predictedClass = equivUsers[0]

        # Whether the target class appears in the equivalence class
        classPredicted = actualClass in equivUsers
        predicted = actualClass == predictedClass

        if verbose:
            print(f"Actual User {actualClass}")
            print(f"Predicted user {predictedClass}")
            print(f"Equivalent Users {equivUsers}")
            print(f"Correct Class Predicted {classPredicted}")
            print(f"Class User Predicted {predicted}\n\n")

        # Counting all the misses to later calculate the class accuracy
        if classPredicted:
            claccuracy += 1
        if predicted:
            indAccuracy += 1

        equivClasses.append(equivUsers)
    
    claccuracy = (claccuracy / classPreds.shape[0]) * 100
    indAccuracy = (indAccuracy / classPreds.shape[0]) * 100

    print("User Percentiles")
    print(f"Class Accuracy: {claccuracy:.2f}%")
    print(f"Individual Accuracy: {indAccuracy:.2f}%\n")

    return claccuracy, indAccuracy, equivClasses

def keyFromValue(myDict, searchValue):
    return list(myDict.keys())[list(myDict.values()).index(searchValue)]

"""
Maps each userID to their corresponding probabilities
"""
def makeUserMap(dataset, predictedProbs):
    filteredIDs = list(filteredMap(dataset=dataset).keys())
    return {userID: prob for userID, prob in zip(filteredIDs, predictedProbs)}