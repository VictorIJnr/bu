#Experimental file to evaluate training accuracy of a model.
#But this isn't any regular old model.
#Oh no.
#This is what I'll call "Restricted Training". 
#In a nutshell this is Dropout. But on a specific set of features at a time.
import os

import pandas as pd

from argparse import ArgumentParser
from collections import defaultdict
from pprint import pprint

from characterisation.helpers.stack import dataPath, serverfault, worldbuilding

from characterisation.classification.sklearnSVM import initSVM, expPredict
from characterisation.classification.sklearnHelper import split
from characterisation.classification.equiv import Equivs

from characterisation.experiments.accies import testWrapper

from helpers import fileIO

savePath = os.path.join(dataPath, "experiments")


def formatCVResults(cvResults):
    #Removing attributes surplus to our accuracy tests.
    del cvResults["mean_fit_time"]
    del cvResults["std_fit_time"]
    del cvResults["mean_score_time"]
    del cvResults["std_score_time"]
    del cvResults["split0_test_score"]
    del cvResults["split1_test_score"]
    del cvResults["split2_test_score"]
    del cvResults["split3_test_score"]
    del cvResults["split4_test_score"]
    del cvResults["split0_train_score"]
    del cvResults["split1_train_score"]
    del cvResults["split2_train_score"]
    del cvResults["split3_train_score"]
    del cvResults["split4_train_score"]

    cvResults = pd.DataFrame(cvResults)
    return cvResults.sort_values("rank_test_score")

def runAccuracyTests(myModel, myDF, mini=False, dataDF=None):
    accResults = []
        
    _, _, xTest, yTest = split(myDF=dataDF)

    for equivTest in testEquivs(myModel, xTest, yTest):
        accResults.append(equivTest)
    
    accResults = pd.DataFrame(accResults)
    return accResults.fillna(0)

def testEquivs(myModel, xTest, yTest):
    equivResults = []

    for equiv in Equivs:
        print(f"Testing the {equiv.name} equivalence class.")
        modelResults = defaultdict(lambda: {})
        testDF = expPredict(myModel, xTest, yTest, equivClass=equiv, individual=True)

        equivResults.append(testDF)

        # #Loop over the equivalence class sizes I guess
        # equivSizes = [len(equivClass) for equivClass in equivClasses]
        # equivSizeCounter = Counter(equivSizes)

        # modelResults["Equivalence Method"] = equiv.name
        # modelResults["User Accuracy"] = indAccuracy
        # modelResults["Class Accuracy"] = claccuracy
        # modelResults["Min Class Size"] = np.amin(equivSizes)
        # modelResults["Max Class Size"] = np.amax(equivSizes)
        # modelResults["Mean Class Size"] = np.mean(equivSizes)
        # modelResults["Largest Possible Class Size"] = len(np.unique(yTest))

        # for i in np.arange(1, np.amax(equivSizes) + 1):
        #     modelResults[f"Class Size {i} Count"] = equivSizeCounter[i]

        # if modelParams is not None:
        #     for paramName, paramValue in modelParams.items():
        #         modelResults[paramName] = paramValue

        # equivResults.append(dict(modelResults))
    
    return equivResults

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
    # createResultsDF(stopSVM, stoppyDF, "StopWordsResults.csv")
    test1Model(stopSVM, stoppyDF, "StopWordsSingleModelResults.csv")

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

def test1Model(myModel, dataDF, fileName):
    resultsDF = formatCVResults(myModel.cv_results_)

    resultsDF = runAccuracyTests(myModel, resultsDF, dataDF=dataDF)

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
        stoppies(myArgs)
