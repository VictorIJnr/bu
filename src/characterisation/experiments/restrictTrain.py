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

from characterisation.classification.sklearnSVM import initSVM, expPredict, predict
from characterisation.classification.sklearnHelper import split
from characterisation.classification.equiv import Equivs

from characterisation.experiments.accies import testWrapper

from helpers import fileIO

#Store every possible permutation on grouped features as possible
Perms = Enum("Perms", "STOPPIES")

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
Run the experiment for all of the features related to the proportion of stop words in the 
users corpus. 
"""
def stoppies(myArgs):
    stopSVM = None
    stoppyDF = loadDataset(myArgs.mini)

    stoppyDF = stoppyDF[list(stoppyDF.filter(regex="stop-|userID"))]

    if not myArgs.load:
        xTrain, yTrain, _, _ = split(myDF=stoppyDF)

        stopSVM = initSVM(xTrain, yTrain, fullSearch=True)

        fileIO.savePickle(stopSVM, "StopWordsSVM.pkl")
    else:
        stopSVM = fileIO.loadPickle("StopWordsSVM.pkl")

    pprint(stopSVM.cv_results_)
    createResultsDF(stopSVM, stoppyDF, "StopWordsResults.csv")
    # test1Model(stopSVM, stoppyDF, featureSet.name + "SingleModelResults.csv")

"""
Creates a DF pertaining to the effectiveness of a model, as determined by metrics.
In particular the individual class accuracy and the equivalence class accuracy. 
"""
def createResultsDF(myModel, dataDF, fileName):
    resultsDF = testWrapper(myModel.cv_results_, splitData=split(myDF=dataDF),
                            fileName=fileName)

    if not os.path.exists(savePath):
        os.makedirs(savePath)

    # pprint(resultsDF)
    resultsDF.to_csv(os.path.join(savePath, fileName))

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
    myParser.add_argument("--mini", "-m", default=False, action="store_true",
                        help="Whether a small subsection of the dataset will be used for training."
                            + "Default: False")
    myParser.add_argument("--load", default=False, action="store_true",
                        help="Whether to load a previously trained model."
                            + "Default: False")

    myArgs = myParser.parse_args()

    if myArgs.stoppies:
        featureSet = Perms.STOPPIES

        stoppies(myArgs)
