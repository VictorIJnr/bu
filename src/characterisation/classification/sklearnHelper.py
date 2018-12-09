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
    
    inputData = None
    userIDs = None

    if miniData:    
        inputData = pd.read_csv(os.path.join(dataPath, dataset, "miniPostsExtracted.csv"))
    else:
        #Pull the full dataset with its extracted features
        pass
    if limitFeatures:
        inputData.drop(["metaFreq", "stopWordFreq"], axis=1, inplace=True)

    userIDs = inputData.pop("userID").values
    inputData = inputData.values

    #Fix this
    trainData, testData, trainIDs, testIDs = train_test_split(inputData, userIDs,
                                                test_size=0.2, train_size=0.8)

    return trainData, trainIDs, testData, testIDs

def hyperSearch(searchModel, paramDist, trainX, trainY, searchNum=20, verbose=True, cv=5):
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

def uniqueUsers():
    inputData = pd.read_csv(os.path.join(dataPath, dataset, "miniPostsExtracted.csv"))
    users = inputData["userID"].values

    np.set_printoptions(suppress=True)
    unique = np.unique(users, return_counts=True)
    retUnique = defaultdict(lambda: 0)

    # print(f"\n\n\n{(type(unique[1]))}\n\n\n")

    #This could probably be resolved into a dict comprehension
    for i in np.arange(unique[0].size):
        retUnique[unique[0][i]] = unique[1][i]
    
    print(f"retUnique items type {type(retUnique.items())}")
    print(retUnique.items())
    print(f"{len(np.unique(users))} unique users")

    return unique[0]