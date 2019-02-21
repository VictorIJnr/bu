import os

import numpy as np
import pandas as pd

from collections import defaultdict
from pprint import pprint
from time import time

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split

from characterisation.helpers.process import dataProcess

worldbuilding = "worldbuilding.stackexchange.com"
serverfault = "serverfault.com"

miniData = True
dataset = worldbuilding

buPath = os.path.dirname(os.path.realpath(__file__))
dataPath = os.path.join(buPath, "..", "..", "..", "data")

"""
Loads the dataset to be used into a dataframe
"""
def __loadData(fullData=False):
    if miniData and not fullData:   
        #Pull a portion of the dataset with extracted features
        inputData = pd.read_csv(os.path.join(dataPath, dataset, "miniRestrictedPostsExtracted.csv"))
    else:
        #Pull the full dataset with its extracted features
        inputData = pd.read_csv(os.path.join(dataPath, dataset, "RestrictedPostsExtracted.csv"))

    return inputData

def pullData(folds=5, fullData=False):
    # I ACTUALLY NEED TO ADD THIS TO THE AUTOENCODER SO I CAN DO DIMENSION REDUCTION FIRST
    # THEN THE RESULT OF THE DIMENSION REDUCTION WILL BE USED HERE IN THE SVM

    #Pull data from csv and yeet it into a df
    #Remove the userID column from the df
    #Use the userID column as the label column
    np.set_printoptions(suppress=True)

    inputData = filterUsers(df=__loadData(fullData))
    
    userIDs = inputData.pop("userID").values.astype(np.uint32)
    inputData = inputData.values

    #Fix this
    #I can't remember what I meant by "Fix this" but I think it's fixed now.
    trainData, testData, trainIDs, testIDs = train_test_split(inputData, userIDs,
                                                test_size=1/folds, train_size=1-(1/folds),
                                                stratify=userIDs)

    return trainData, trainIDs, testData, testIDs

"""
Similar to its private equivalent, loads a dataset into a dataframe.

The difference between this and the private version is the specification of a dataset.
This is meant to be used in tamden with other files, like "process.py" when automating
a certain part of bu. Like processing a new dataset or using a dataset to train the
dimension reduction ConvNet.
"""
def loadData(myDataset="worldbuilding", mini=True):
    #I REALLY NEED TO LEARN PYTHON ENUMS SOON
    if myDataset == "worldbuilding":
        dataset = worldbuilding
    elif myDataset == "serverfault":
        dataset = serverfault

    if mini:   
        #Pull a portion of the dataset with extracted features
        try:
            inputData = pd.read_csv(os.path.join(dataPath, dataset, "miniRestrictedPostsExtracted.csv"))
        except FileNotFoundError:
            #Iff the file doesn't exist, create it and load the data again
            dataProcess(myDataset, "Posts", 1024)
            return loadData(myDataset, mini)
    else:
        #Pull the full dataset with its extracted features
        try:
            inputData = pd.read_csv(os.path.join(dataPath, dataset, "RestrictedPostsExtracted.csv"))
        except FileNotFoundError:
            dataProcess(myDataset, "Posts")
            return loadData(myDataset, mini)

    inputData.drop("postID", axis=1, inplace=True)
    inputData.drop("Unnamed: 0", axis=1, inplace=True)
    return inputData

"""
As loadData() is to __loadData(), split() is the same for pullData()
Essentially, pullData() is the development version, while splitData() is for production. 
I guess?

Anyway, this is used for automating stuff and being adaptable to a provided dataset.
Just like __loadData(), pullData() will be kept for backwards compatability.
I should eventually come up with a better name than "split()" though, I can't use pullData()
yet because a bunch of other development stuff will break.
"""
def split(myDataset="worldbuilding", mini=True, folds=5):
    if myDataset == "worldbuilding":
        dataset = worldbuilding
    elif myDataset == "serverfault":
        dataset = serverfault

    inputData = filterUsers(df=loadData(myDataset, mini))

    print("Input Data Keys in split")
    pprint(inputData.keys())
    
    userIDs = inputData.pop("userID").values.astype(np.uint32)
    inputData = inputData.values
    

    print(f"inputData.shape in split\n{inputData.shape}")

    trainData, testData, trainIDs, testIDs = train_test_split(inputData, userIDs,
                                                test_size=1/folds, train_size=1-(1/folds),
                                                stratify=userIDs)

    return trainData, trainIDs, testData, testIDs

def hyperSearch(searchModel, paramDist, trainX, trainY, searchNum=20, verbose=True, cv=5):
    model = RandomizedSearchCV(searchModel, param_distributions=paramDist, n_iter=searchNum, 
                                cv=cv, verbose=2 if verbose else 0)
    
    print("Training model...")

    start = time()
    model.fit(trainX, trainY)

    #Only keep this whilst I have all those stupid warnings
    #Some of the warnings aren't stupid, like the one warning me about the number of folds
    print("")

    if verbose:
        print(f"Parameter search took {time() - start:.2f} seconds to explore {searchNum}"
            + " possibilities")
        print(len(model.cv_results_))
        report(model.cv_results_)

    return model


"""
Runs an exhaustive search through all the paramaters specified in the parameter distribution
"""
def fullHyperSearch(searchModel, paramDist, trainX, trainY, verbose=True, cv=5):
    model = GridSearchCV(searchModel, param_grid=paramDist, cv=cv, verbose=2 if verbose else 0)

    start = time()
    model.fit(trainX, trainY)
    
    if verbose:
        print(f"Parameter search took {time() - start:.2f} seconds to explore {len(model.cv_results_)}"
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

"""
Filters users who have more than 'threshold' occurences in the dataset
Specifying threshold should be done in accordance to 'folds' in pullData
'threshold' should always be equal to 'folds' 
"""
def filterUsers(df=None, threshold=5):
    if df is None:
        df = __loadData()
    
    threshMap = filteredMap(df, threshold=threshold)
    threshold += 1

    # print("Threshold Users")
    # pprint(threshMap)

    users = np.unique(df["userID"].values.astype(np.uint32))

    #Altering the userID column to only keep users who have met the threshold 
    users = np.delete(users, np.where(np.in1d(users, list(threshMap.keys()), assume_unique=True, invert=True)))

    #Filtering the old data such that only users with a sufficient amount of records are present
    newDF = df[df["userID"].isin(users)]
    # print("New dataframe")
    # pprint(newDF)

    filtered = np.unique(newDF["userID"].values.astype(np.uint32), return_counts=True)
    filtered = {userID: count for userID, count in zip(filtered[0], filtered[1])}

    return newDF

"""
Creates and returns a dictionary of users passing a given threshold
The supplied threshold, which should always be equal to 'folds'
"""
def filteredMap(df=None, dataset=None, threshold=5):
    if df is None:
        df = __loadData()
    if dataset is not None:
        pass

    threshold += 1

    users = df["userID"].values.astype(np.uint32)
    unique = np.unique(users, return_counts=True)
    newDF = None

    #It took me way too long to make this line of code. UGH.
    #Just a dictionary of keys to their frequency in the dataset, where each key is a userID
    threshMap = {int(userID): count for userID, count in zip(unique[0], unique[1])}

    #Filtering the userID-frequency mapping to only get users with enough posts
    mapCopy = threshMap.copy()
    for userID, count in mapCopy.items():
        if count < threshold:
            del threshMap[userID]

    return threshMap

"""
I don't remember what this does and I've got a filtering bug I need
to sort out before I figure out what's going on here
"""
def uniqueUsers():
    inputData = pd.read_csv(os.path.join(dataPath, dataset, "miniRestrictedPostsExtracted.csv"))
    users = inputData["userID"].values

    unique = np.unique(users, return_counts=True)
    retUnique = defaultdict(lambda: 0)

    #This could probably be resolved into a dict comprehension
    for i in np.arange(unique[0].size):
        retUnique[unique[0][i]] = unique[1][i]
    
    return unique[0].astype(np.uint32)