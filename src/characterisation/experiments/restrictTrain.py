#Experimental file to evaluate training accuracy of a model.
#But this isn't any regular old model.
#Oh no.
#This is what I'll call "Restricted Training". 
#In a nutshell this is Dropout. But on a specific set of features at a time.
import os

import pandas as pd

from argparse import ArgumentParser
from pprint import pprint

from characterisation.helpers.stack import dataPath, serverfault, worldbuilding

from characterisation.classification.sklearnSVM import initSVM
from characterisation.classification.sklearnHelper import split

from characterisation.experiments.accies import testWrapper

from helpers import fileIO

savePath = os.path.join(dataPath, "experiments")


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
    stoppyDF = loadDataset(myArgs.mini)

    stoppyDF = stoppyDF[list(stoppyDF.filter(regex="stop-|userID"))]

    xTrain, yTrain, _, _ = split(myDF=stoppyDF)

    stopSVM = initSVM(xTrain, yTrain, fullSearch=True)

    fileIO.savePickle(stopSVM, "StopWordsSVM.pkl")

    pprint(stopSVM.cv_results_)
    createResultsDF(stopSVM, stoppyDF, "StopWordsResults.csv")

"""
Creates a DF pertaining to the effectiveness of a model, as determined by metrics.
In particular the individual class accuracy and the equivalence class accuracy. 
"""
def createResultsDF(myModel, dataDF, fileName):
    resultsDF = testWrapper(myModel.cv_results_, dataDF=dataDF)

    if not os.path.exists(savePath):
        os.makedirs(savePath)

    pprint(resultsDF)
    resultsDF.to_csv(os.path.join(savePath, fileName))

if __name__ == "__main__":
    myParser = ArgumentParser()

    myParser.add_argument("--stoppies", default=False, action="store_true",
                        help="Train the subset of features pertaining to stop words."
                            + "Default: False")
    myParser.add_argument("--mini", "-m", default=False, action="store_true",
                        help="Whether a small subsection of the dataset will be used for training."
                            + "Default: False")

    myArgs = myParser.parse_args()

    if myArgs.stoppies:
        stoppies(myArgs)