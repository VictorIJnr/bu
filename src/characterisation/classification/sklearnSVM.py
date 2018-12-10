import numpy as np


from pprint import pprint

from . import sklearnHelper as skh
from .sklearnHelper import hyperSearch
from .sklearnHelper import pullData, filteredMap
from .equiv import jumpy

from sklearn.svm import SVC

"""
This should retrieve the top x models found from the searching cross-validation
These models will the all be tested against the different equivalence class methods
(Jump Points, 95th Percentile of Users, Highest 90% of Scores)
"""
def pullTopX(num=5):
    pass

def initSVM():
    trainX, trainY, testX, testY = pullData()

    print(f"{len(np.unique(trainY))} different training classes\n\n")

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
    classy = hyperSearch(SVC(probability=True), paramDist, trainX, trainY, searchNum=5)
    # classy = hyperSearch(SVC(), paramDist, trainX, trainY, searchNum=100)

    # print(type(classy.cv_results_))
    # pprint(classy.cv_results_)

    skh.saveModel(classy, "classy.pkl")

    #This is a bad way of retaining the test data... I'll find another way to fix this sometime
    #Huh, I just thought of one, pass the training parameters to this method instead
    #So outside the method you still have the train/test split that you used.
    return classy, testX, testY

"""
Runs experiments on each of the different methods to determine equivalence classes
"""
def testEquiv():
    classy, testX, testY = initSVM()
    predictProbs = classy.predict_proba(testX)
    predictions = classy.predict(testX)

    mapping = filteredMap()
    print(f"Predictions: {predictions}")
    print(f"Prediction Probabilities: {predictProbs}")
    print(f"Actual Values: {testY}")

    jumpy(predictProbs, testY)

if __name__ == "__main__":
    testEquiv()