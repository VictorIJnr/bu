#Experimental file to evaluate training accuracy of a model.
#But this isn't any regular old model.
#Oh no.
#This is what I'll call "Restricted Training". 
#In a nutshell this is Dropout. But on a specific set of features at a time.
import os

import pandas as pd

from argparse import ArgumentParser
from collections import defaultdict
from enum import Enum
from pprint import pprint

from characterisation.helpers.stack import dataPath, serverfault, worldbuilding

from characterisation.classification.sklearnSVM import initSVM
from characterisation.classification.sklearnHelper import split
from characterisation.classification.equiv import Equivs

from characterisation.experiments.accies import probWrapper, testWrapper

from helpers import fileIO

#Store every possible permutation on grouped features as possible
Perms = Enum("Perms", "STOPPIES RICHNESS AVERAGES STOP_RICH STOP_AVG RICH_AVG STOP_RICH_AVG")

savePath = os.path.join(dataPath, "experiments")

# Indicates the permutation of features being used
featureSet = None

# Need to create a DF which shows the accuracy related to each equivalence class,
# For each prediction, show the equivalence class size, the predicted equivalence class,
# the actual class, and a boolean of whether the predicted class was within the equiv class

"""
Loads the base dataset to be used for training and evaluating a model.
The base dataset, containing the complete 303 features 
"""
def loadDataset(mini):
    if mini:
        return pd.read_csv(os.path.join(dataPath, worldbuilding, "miniRestrictedPostsExtracted.csv"))
    else:
        return pd.read_csv(os.path.join(dataPath, worldbuilding, "RestrictedPostsExtracted.csv"))

"""
Just setting a global to determine which subset of features to be used for training
"""
def determineFeatureSet(myArgs):
    global featureSet

    if myArgs.stoppies:
        featureSet = Perms.STOPPIES
        if myArgs.richness:
            featureSet = Perms.STOP_RICH 
            if myArgs.averages:
                featureSet = Perms.STOP_RICH_AVG 
        if myArgs.averages:
            featureSet = Perms.STOP_AVG 
    elif myArgs.richness:
        featureSet = Perms.RICHNESS
        if myArgs.averages:
            featureSet = Perms.RICH_AVG 
    elif myArgs.averages:
        featureSet = Perms.AVERAGES


"""
Returns the corresponding regex for the features for each possible permutation
"""
def getPermRegex():
    averageRegex = "^num|^avg|userID"
    richnessRegex = "lego|yule|userID"
    stoppiesRegex = "stop-|userID"

    if featureSet == Perms.STOPPIES:
        return stoppiesRegex
    elif featureSet == Perms.RICHNESS:
        return richnessRegex
    elif featureSet == Perms.AVERAGES:
        return averageRegex
    elif featureSet == Perms.STOP_RICH:
        return f"{stoppiesRegex}|{richnessRegex}"
    elif featureSet == Perms.STOP_AVG:
        return f"{stoppiesRegex}|{averageRegex}"
    elif featureSet == Perms.RICH_AVG:
        return f"{richnessRegex}|{averageRegex}"
    elif featureSet == Perms.STOP_RICH_AVG:
        return f"{stoppiesRegex}|{richnessRegex}|{averageRegex}"

"""
Wrapper for the restrictTrain method.
Allows restrictTrain() to be called within the code to generate probability files
for each possible permutation of feature set groups
"""
def restrictWrapper(myArgs):
    determineFeatureSet(myArgs)

    print(featureSet)
    restrictTrain(myArgs.load, myArgs.mini)

"""
Runs the restrictive training experiment for a determined feature set on a user's corpus.
"""
def restrictTrain(load=False, mini=False):
    mySVM = None
    myDF = loadDataset(mini)
    myDF = myDF[list(myDF.filter(regex=getPermRegex()))]
    
    # Ensure that the train and test sets are distinct
    splitData = split(myDF=myDF)

    if not load:
        xTrain, yTrain, _, _ = splitData

        # Training a new SVM model and saving it to a file
        mySVM = initSVM(xTrain, yTrain, fullSearch=True)
        fileIO.savePickle(mySVM, f"{featureSet.name.title()}WordsSVM.pkl")
    else:
        mySVM = fileIO.loadPickle(f"{featureSet.name.title()}WordsSVM.pkl")

    createProbsCSV(mySVM, splitData, f"{featureSet.name.title()}WordsSVM.csv")

"""
Runs the accuracy experiments for the calculated probabilities
for each set of feature set permutations
"""
def runAcciesExp():
    pass

"""
Creates a DF pertaining to the effectiveness of a model, as determined by metrics.
In particular the individual class accuracy and the equivalence class accuracy. 
"""
def createProbsCSV(myModel, splitData, fileName):
    resultsDF = probWrapper(myModel.cv_results_, splitData=splitData,
                            fileName=fileName)

    if not os.path.exists(savePath):
        os.makedirs(savePath)

    # pprint(resultsDF)
    resultsDF.to_csv(os.path.join(savePath, fileName))

def createAcciesCSV():
    pass

"""
Creates the CSV for predicted probabilities for each of the different feature set permutations.
Accompanies the autoRestrict() method to complete the experiment.
"""
def autoProbsCSV(myArgs):
    global featureSet

    for perm in Perms:
        featureSet = perm
        restrictTrain(myArgs.load, myArgs.mini)

"""
Run the restriction experiment on all of the generated probability files.
"""
def autoRestrict():
    global featureSet

    for perm in Perms:
        featureSet = perm
        restrictTrain(myArgs.load, myArgs.mini)

"""
Runs the experiment against only the best model found from the Cross-Validation search.
"""
def test1Model(myModel, dataDF, fileName="SingleModelRestriction.csv"):
    resultsDF = testWrapper(myModel.cv_results_, splitData=split(myDF=dataDF), 
                            fileName=fileName, single=True)

    if not os.path.exists(savePath):
        os.makedirs(savePath)

    resultsDF.to_csv(os.path.join(savePath, fileName))

if __name__ == "__main__":
    myParser = ArgumentParser()

    myParser.add_argument("--stoppies", default=False, action="store_true",
                        help="Train the subset of features pertaining to stop words."
                            + "Default: False")
    myParser.add_argument("--richness", default=False, action="store_true",
                        help="Train the subset of features pertaining to vocabulary richness."
                            + "Default: False")
    myParser.add_argument("--averages", default=False, action="store_true",
                        help="Train the subset of features pertaining to averages in a user's corpus."
                            + "Default: False")
    myParser.add_argument("--mini", "-m", default=False, action="store_true",
                        help="Whether a small subsection of the dataset will be used for training."
                            + "Default: False")
    myParser.add_argument("--load", default=False, action="store_true",
                        help="Whether to load a previously trained model."
                            + "Default: False")

    myArgs = myParser.parse_args()

    determineFeatureSet(myArgs)

    restrictWrapper(myArgs)
