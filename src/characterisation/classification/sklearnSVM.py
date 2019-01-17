import numpy as np


from pprint import pprint

from helpers import fileIO
from .sklearnHelper import hyperSearch
from .sklearnHelper import pullData, filteredMap
from .equiv import jumpy, userCentiles

from sklearn.svm import SVC

"""
This should retrieve the top x models found from the searching cross-validation
These models will the all be tested against the different equivalence class methods
(Jump Points, 95th Percentile of Users, Highest 90% of Scores)
"""
def pullTopX(num=5):
    pass

def initSVM(trainX, trainY, loadModel=False):
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
    #Run that search in the background of a lab machine
    #That calculation is actually wrong, I haven't recalculated it but it'll be MUCH longer
    if loadModel:
        classy = loadSVM()
    else:
        classy = hyperSearch(SVC(probability=True), paramDist, trainX, trainY, searchNum=5)
        # classy = hyperSearch(SVC(), paramDist, trainX, trainY, searchNum=100)

        # print(type(classy.cv_results_))
        # pprint(classy.cv_results_)

        fileIO.savePickle(classy, "classy.pkl")

    return classy

"""
Loads a previously trained SVM model
"""
def loadSVM():
    return fileIO.loadPickle("classy.pkl")

"""
Runs experiments on each of the different methods to determine equivalence classes
"""
def testEquiv():
    trainX, trainY, testX, testY = pullData()
    classy = initSVM(trainX, trainY, True)

    predictProbs = classy.predict_proba(testX)
    predictions = classy.predict(testX)

    mapping = filteredMap()

    jumpy(predictProbs, testY)
    userCentiles(predictProbs, testY)

if __name__ == "__main__":
    testEquiv()