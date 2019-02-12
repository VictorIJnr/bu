import numpy as np

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
"""
def initSVM(trainX, trainY, loadModel=False, searchNum=5, fullSearch=False):
    print(f"{len(np.unique(trainY))} different training classes\n\n")

    paramDist = {
        # "kernel": ["rbf", "sigmoid"],
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
        try:
            classy = loadSVM()
        except:
            return initSVM(trainX, trainY, searchNum=searchNum, fullSearch=fullSearch)
    else:
        if fullSearch:
            classy = fullHyperSearch(SVC(probability=True), paramDist, trainX, trainY, searchNum=searchNum)
            fileIO.savePickle(classy, f"classySVM_FullSearch.pkl")
        else:
            classy = hyperSearch(SVC(probability=True), paramDist, trainX, trainY, searchNum=searchNum)
            # classy = hyperSearch(SVC(), paramDist, trainX, trainY, searchNum=100)

            # print(type(classy.cv_results_))
            # pprint(classy.cv_results_)

            fileIO.savePickle(classy, f"classySVM_{searchNum}Searches.pkl")

    return classy

"""
Predicts the equivalence class of users, with the provided algorithm given a model

It's easier to use this method by passing a model instead of expecting to pass
very specific data.
"""
def predict(model, xInput, equivClass=Equivs.JUMP, dataset="worldbuilding"):
    probs = model.predict_proba(xInput)

    if equivClass == Equivs.JUMP:
        return jumpy(dataset, probs)
    elif equivClass == Equivs.SCORE_DIST:
        return scoreDistri(dataset, probs)
    elif equivClass == Equivs.PERCENTILES:
        return userCentiles(dataset, probs)

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

    predictProbs = classy.predict_proba(testX)
    predictions = classy.predict(testX)

    mapping = filteredMap()

    scoreDistriExperimental(predictProbs, testY, dataset="worldbuilding")
    
    jumpyExperimental(predictProbs, testY, dataset="worldbuilding")
    userCentilesExperimental(predictProbs, testY, dataset="worldbuilding")

if __name__ == "__main__":
    myParser = ArgumentParser()

    myParser.add_argument("--load", default=False, action="store_true")
    myArgs = myParser.parse_args()

    testEquiv(loadModel=myArgs.load)