import numpy as np

import characterisation.classification.sklearnHelper as skh

from argparse import ArgumentParser
from pprint import pprint

from helpers import fileIO
from characterisation.classification.sklearnHelper import hyperSearch, fullHyperSearch
from characterisation.classification.sklearnHelper import pullData, filteredMap
from characterisation.classification.equiv import jumpy, userCentiles, scoreDistri, Equivs
from characterisation.classification.equiv import jumpyExperimental, userCentilesExperimental, scoreDistriExperimental

from sklearn.svm import SVC

worldbuilding = "worldbuilding.stackexchange.com"
serverfault = "serverfault.com"

"""
This should retrieve the top x models found from the searching cross-validation
These models will the all be tested against the different equivalence class methods
(Jump Points, 95th Percentile of Users, Highest 90% of Scores)
"""
def pullTopX(num=5):
    pass

"""
Trains a SVM for user classification
Performs hyper-parameter optimisation in the process
"""
def initSVM(trainX, trainY, paramDist=None, loadModel=False, searchNum=5, fullSearch=False, verbose=True):
    print(f"{len(np.unique(trainY))} different training classes\n")

    if paramDist is None:
        paramDist = {
            "kernel": ["rbf", "sigmoid"],
            # "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "degree": list(range(6)),
            "gamma": ["auto", "scale", 0.01, 0.05, 0.1, 0.15, 0.2],
            "coef0": np.linspace(0, 1, num=21), #21 to accomodate for the endpoint (1)
            "shrinking": [True, False],
            "tol": [0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
        }

        kernelDist = {
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            # "coef0": np.linspace(0, 1, num=21),
            # "gamma": np.linspace(0.1, 1, num=19)
             "gamma": ["scale"]
        }

        paramDist = kernelDist

    if loadModel:
        try:
            classy = loadSVM()
        except:
            # Create a new SVM from scratch
            return initSVM(trainX, trainY, searchNum=searchNum, fullSearch=fullSearch, verbose=verbose)
    else:
        if fullSearch:
            classy = fullHyperSearch(SVC(probability=True), paramDist, trainX, trainY, verbose=verbose)
            fileIO.savePickle(classy, f"classySVM_FullSearch.pkl")
        else:
            classy = hyperSearch(SVC(probability=True), paramDist, trainX, trainY, 
                                    searchNum=searchNum, verbose=verbose)
            fileIO.savePickle(classy, f"classySVM_{searchNum}Searches.pkl")

    return classy

"""
Predicts the user directly from the SVM input. Does not use equiv classes.
Solely returns the probabilities for each of the trained user classes
"""
def predictProbs(model, xInput):
    return skh.predictProbs(model, xInput)

"""
Predicts the equivalence class of users, with the provided algorithm given a model

It's easier to use this method by passing a model instead of expecting to pass
very specific data.
"""
def predict(model, xInput, equivClass=Equivs.JUMP, dataset="worldbuilding",
                returnProbs=False, probs=None):
    return skh.predict(model, xInput, equivClass, dataset, returnProbs, probs)

"""
Experimental method to predict the equivalence class of users

Unlike the non-experimental version, this uses the experimental equiv class
methods to return the accuracies instead of the predicted class.
"""
def expPredict(model, xTest, yTest, equivClass=Equivs.JUMP, dataset="worldbuilding", individual=False,
                returnProbs=False):
    return skh.expPredict(model, xTest, yTest, equivClass, dataset, returnProbs)

"""
Loads a previously trained SVM model
"""
def loadSVM():
    return fileIO.loadPickle("classySVM.pkl")

"""
Runs experiments on each of the different methods to determine equivalence classes
"""
def testEquiv(loadModel=False):
    trainX, trainY, testX, testY = pullData()
    classy = initSVM(trainX, trainY, loadModel=loadModel)

    probs = classy.predict_proba(testX)
    predictions = classy.predict(testX)

    mapping = filteredMap()

    scoreDistriExperimental(probs, testY, dataset="worldbuilding")
    
    jumpyExperimental(probs, testY, dataset="worldbuilding")
    userCentilesExperimental(probs, testY, dataset="worldbuilding")

if __name__ == "__main__":
    myParser = ArgumentParser()

    myParser.add_argument("--load", default=False, action="store_true")
    myArgs = myParser.parse_args()

    testEquiv(loadModel=myArgs.load)
