import os

import numpy as np
import pandas as pd

from pprint import pprint

from .sklearnHelper import hyperSearch, report

from sklearn.model_selection import train_test_split
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
    
    inputData = None
    userIDs = None

    if miniData:    
        inputData = pd.read_csv(os.path.join(dataPath, dataset, "miniPostsExtracted.csv"))
    else:
        pass
    if limitFeatures:
        inputData.drop(["metaFreq", "stopWordFreq"], axis=1, inplace=True)

    userIDs = inputData.pop("userID").values
    inputData = inputData.values

    #Fix this
    trainData, testData, trainIDs, testIDs = train_test_split(inputData, userIDs,
                                                test_size=0.2, train_size=0.8)

    return trainData, trainIDs, testData, testIDs

"""
This should retrieve the top x models found from the searching cross-validation
These models will the all be tested against the different equivalence class methods
(Jump Points, 95th Percentile of Users, Highest 90% of Scores)
"""
def pullTopX(num=5):
    pass

def initSVM():
    trainX, trainY, testX, testY = pullData()

    paramDist = {
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": list(range(6)),
        "gamma": ["auto", "scale", 0.01, 0.05, 0.1, 0.15, 0.2],
        "coef0": np.linspace(0, 1, num=21), #21 to accomodate for the endpoint (1)
        "shrinking": [True, False],
        "tol": [0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    }

    #I calculated it, doing a complete search with all of these parameters will take 
    #5 and a half days...
    classy = hyperSearch(SVC(), paramDist, trainX, trainY, searchNum=5)
    # classy = hyperSearch(SVC(), paramDist, trainX, trainY, searchNum=100)

    # print(type(classy.cv_results_))
    # pprint(classy.cv_results_)

    #This is a bad way of retaining the test data... I'll find another way to fix this sometime
    #Huh, I just thought of one, pass the training parameters to this method instead
    #So outside the method you still have the train/test split that you used.
    return classy, testX, testY

"""
Runs experiments on each of the different methods to determine equivalence classes
"""
def testEquiv():
    classy, testX, testY = initSVM()

    classy.predict(testX)

if __name__ == "__main__":
    testEquiv()