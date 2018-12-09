import os

import numpy as np
import pandas as pd

from collections import defaultdict
from pprint import pprint
from time import time

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

worldbuilding = "worldbuilding.stackexchange.com"
serverfault = "serverfault.com"

miniData = True
limitFeatures = True
dataset = worldbuilding

buPath = os.path.dirname(os.path.realpath(__file__))
dataPath = os.path.join(buPath, "..", "..", "..", "data")

def pullData():
    # I ACTUALLY NEED TO ADD THIS TO THE AUTOENCODER SO I CAN DO DIMENSION REDUCTION FIRST
    # THEN THE RESULT OF THE DIMENSION REDUCTION WILL BE USED HERE IN THE SVM

    #Pull data from csv and yeet it into a df
    #Remove the userID column from the df
    #Use the userID column as the label column
    np.set_printoptions(suppress=True)

    inputData = None
    userIDs = None

    if miniData:    
        inputData = pd.read_csv(os.path.join(dataPath, dataset, "miniPostsExtracted.csv"))
    else:
        #Pull the full dataset with its extracted features
        pass
    if limitFeatures:
        inputData.drop(["metaFreq", "stopWordFreq"], axis=1, inplace=True)

    inputData = filterUsers(inputData)
    
    userIDs = inputData.pop("userID").values.astype(np.uint32)
    inputData = inputData.values

    #Fix this
    #I can't remember what I meant by "Fix this" but I think it's fixed now.
    trainData, testData, trainIDs, testIDs = train_test_split(inputData, userIDs,
                                                test_size=0.2, train_size=0.8,
                                                stratify=userIDs)

    return trainData, trainIDs, testData, testIDs

def hyperSearch(searchModel, paramDist, trainX, trainY, searchNum=20, verbose=True, cv=2):
    model = RandomizedSearchCV(searchModel, param_distributions=paramDist, n_iter=searchNum, cv=cv)
    
    start = time()
    model.fit(trainX, trainY)

    print("") #Only keep this whilst I have all those stupid warnings

    if verbose:
        print(f"Parameter search took {time() - start:.2f} seconds to explore {searchNum}"
            + " possibilities")
        report(model.cv_results_)

    return model


"""
Runs an exhaustive search through all the paramaters specified in the parameter distribution
"""
def fullHyperSearch(searchModel, paramDist, trainX, trainY, verbose=True, cv=5):
    model = GridSearchCV(searchModel, param_distributions=paramDist, cv=cv)

    start = time()
    model.fit(trainX, trainY)
    
    if verbose:
        print(f"Parameter search took {time() - start:.2f} seconds to explore {searchNum}"
            + " possibilities")
        report(model.cv_results_)
    
    return model

def report(results, bestN=3):
    for i in range(1, bestN + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print(f"Model with rank: {i}")
            print(f"Mean validation score: {results['mean_test_score'][candidate]:.3f} "
                + f"(std: {results['std_test_score'][candidate]:.3f})")
            print(f"Parameters: {results['params'][candidate]}\n")

def filterUsers(dataframe):
    users = dataframe["userID"].values.astype(np.uint32)
    unique = np.unique(users, return_counts=True)
    newDF = None

    #It took me way too long to make this line of code. UGH.
    #Just a dictionary of keys to their frequency in the dataset, where each key is a userID
    threshMap = {int(userID): count for userID, count in zip(unique[0], unique[1])}

    #Filtering the userID-frequency mapping to only get users with enough posts
    mapCopy = threshMap.copy()
    for userID, count in mapCopy.items():
        if count < 5:
            del threshMap[userID]

    print("Threshold Users")
    pprint(threshMap)

    #Creating a list of all the 
    users = np.delete(users, np.where(np.in1d(users, list(threshMap.keys()), assume_unique=True, invert=True)))

    #Filtering the old data such that only users with a sufficient amount of records are present
    newDF = dataframe[dataframe["userID"].isin(users)]
    print("New dataframe")
    pprint(newDF)

    return newDF

def uniqueUsers():
    inputData = pd.read_csv(os.path.join(dataPath, dataset, "miniPostsExtracted.csv"))
    users = inputData["userID"].values

    unique = np.unique(users, return_counts=True)
    retUnique = defaultdict(lambda: 0)

    #This could probably be resolved into a dict comprehension
    for i in np.arange(unique[0].size):
        retUnique[unique[0][i]] = unique[1][i]
    
    print(f"retUnique items type {type(retUnique.items())}")
    print(retUnique.items())
    print(f"{len(np.unique(users))} unique users")

    return unique[0].astype(np.uint32)