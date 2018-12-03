import os

import numpy as np
import pandas as pd

from pprint import pprint

from .sklearnHelper import hyperSearch, report

from sklearn.svm import SVC

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
    
    trainData = None
    userIDs = None

    if miniData:    
        trainData = pd.read_csv(os.path.join(dataPath, dataset, "miniPostsExtracted.csv"))
    else:
        pass
    
    if limitFeatures:
        trainData.drop(["metaFreq", "stopWordFreq"], axis=1, inplace=True)

    userIDs = trainData.pop("userID").values
    trainData = trainData.values

    print(type(userIDs))
    print(type(trainData))
    return trainData, userIDs

def initSVM():
    trainX, trainY = pullData()

    print(trainX.shape)
    print(trainY.shape)

    paramDist = {
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": list(range(6)),
        "gamma": ["auto", "scale", 0.01, 0.05, 0.1, 0.15, 0.2],
        "coef0": np.linspace(0, 1, num=20),
        "shrinking": [True, False],
        "tol": [0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    }

    #I calculated it, doing a complete search with all of these parameters will take 
    #5 and a half days...
    classy = hyperSearch(SVC(), paramDist, trainX, trainY, searchNum=5)
    # classy = hyperSearch(SVC(), paramDist, trainX, trainY, searchNum=100)

    print(type(classy.cv_results_))
    pprint(classy.cv_results_)

if __name__ == "__main__":
    initSVM()