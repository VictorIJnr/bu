#Experiment file testing against the accuracies of different models 
import os
import re

# pylint: disable=import-error
import seaborn as sns
import pandas as pd
import numpy as np 

import characterisation.classibu as classibu

from argparse import ArgumentParser
from collections import Counter, defaultdict
from pprint import pprint

from matplotlib import pyplot as plt
from sklearn.model_selection import RandomizedSearchCV

from characterisation.classibu import skippedSVM, svmPredict, rawSVMPredict
from characterisation.classification.equiv import Equivs 
from characterisation.classification.sklearnSVM import expPredict, initSVM 
from characterisation.classification.sklearnHelper import split, filteredMap

# pylint: disable=no-name-in-module
from helpers import fileIO

"""
Wraps the functionality of performing accuracy tests into one method dependant
on having Cross-Validation results.

:param probaFile the name of the file to retrieve calculated probabilities
:param fileName the name of the file to save the dataframe containing the results
"""
def testWrapper(probaFile="ExpProbabilities.csv", fileName="AcciesResults.csv"):
    return runAccuracyTests(probaFile=probaFile, fileName=fileName)

"""
Wraps the functionality for generating the CSV for the probability distribution of users.
"""
def probWrapper(cvResults, mini=False, splitData=None, fileName="ExpProbabilities.csv", single=False):
    resultsDF = formatCVResults(cvResults)

    return runPredictions(resultsDF, mini, splitData, fileName=fileName, single=single)

"""
Formats the results for a previous Cross-Validation search over a distribution of 
hyper-parameters. Only keeping attributes which are required for subsequent
accuracy tests.
"""
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

"""
Runs predictions for the provided dataset against a series of provided models.
"""
def runPredictions(myDF, mini=False, splitData=None, fileName="", single=False):
    preds = []
    cleanupFiles = []

    loopIters = 0
    for _, model in myDF.iterrows(): 
        if single and loopIters == 1:
            break

        intermediateFile = fileName[:-4] + model["params"]["kernel"] + ".csv"

        # Just saving dataframes as we go along so we don't lose anything
        preds.extend(predictTests(model, splitData, mini, fileName))
        pd.DataFrame(preds).to_csv(intermediateFile)
        
        cleanupFiles.append(intermediateFile)
        
        loopIters += 1

    for cleanFile in cleanupFiles:
        os.remove(cleanFile)

    preds = pd.DataFrame(preds)
    preds.to_csv(fileName)

    return preds

"""
Calculates the probabilities for each of the trained user classes for each 
test instance (either provided or generated automatically)
"""
def predictTests(model, splitData, mini, fileName):
    print("\nCurrently calculating predictions against this model:")
    pprint(model)

    myModel = None
    xTest, yTest = None, None
    paramDist = {paramName: [paramValue] for paramName, paramValue in model["params"].items()}
    
    if splitData is None:
        myModel, xTest, yTest = skippedSVM(paramDist=paramDist, mini=mini, searchNum=1,
                                    returnTest=True, verbose=False)
    else:
        xTrain, yTrain, xTest, yTest = splitData
        myModel = initSVM(xTrain, yTrain, paramDist=paramDist, fullSearch=True)
    
    return calcTestProbs(myModel, xTest, yTest)

"""
Calculates the probabilities of the entire provided test instance.

Creates a list of dictionaries for each of the sampled test instances to 
later be stored in a DF which can be used for further processing.
Either way the results will be stored into a CSV for later use.
"""
def calcTestProbs(myModel, xTest, yTest):
    predResults = []

    for i in np.arange(len(xTest)):
        predicty = defaultdict(lambda: None)
        userIDs = list(filteredMap().keys())

        testInput = xTest[i]
        actualClass = yTest[i]
        
        predProbs = rawSVMPredict(myModel, testInput)
        predClass = userIDs[np.argmax(predProbs)]
        
        predicty.update({f"User ID {userID}": prob for userID, prob in zip(userIDs, predProbs)})

        predicty["Actual Class"] = actualClass
        predicty["Predicted Class"] = predClass
        predicty["User Predicted"] = actualClass == predClass

        # print(f"LOOK AT ALL THESE KEYS {myModel.get_params()['cv']}")
        # print(f"LOOK AT ALL THESE KEYS {type(myModel)}")
        # print(f"LOOK AT ALL THESE KEYS {myModel.get_params().keys()}")
        # print(f"LOOK AT ALL THESE KEYS {myModel.get_params()['param_grid']}")
        # print(f"LOOK AT ALL THESE KEYS {myModel.get_params()['estimator']}")

        modelParams = myModel.get_params()["param_distributions"] \
            if isinstance(myModel, RandomizedSearchCV) \
            else myModel.get_params()["param_grid"]

        predicty.update({param: paramVals[0] for param, paramVals 
            in modelParams.items()})

        predResults.append(predicty)

    return predResults

"""
Runs tests against the accuracy of models

@param single indicates whether to test only the best performing model,
    determined by the Cross-Validation
"""
def runAccuracyTests(probaFile="ExpProbabilities.csv", fileName=""):
    probaDF = pd.read_csv(probaFile)
    probaDF.drop(list(probaDF.filter(regex="Unnamed*")), axis=1, inplace=True)

    results = testEquivs(probaDF)
    results = pd.DataFrame(results)

    results.to_csv(fileName)

    return results

"""
Testing against all of the different equivalence classes
Allows for evaluation using the generated probabilities CSV

The incremental equivs method should become the de facto method and I'll deprecate the 
other one sometime soon.
"""
def testEquivs(probaDF):
    equivResults = []

    newProbs, probaDF = formatProbs(probaDF)
    numRows = len(probaDF.index)

    for i in np.arange(numRows):
        print(f"Testing index {i + 1} of "
               + f"{numRows} - {((i + 1)/ numRows) * 100:.3f}% done.",
                end="\r")
        
        # Even though I'm using already predicted probabilities, I still need each Equiv class
        for equiv in Equivs:
            
            # Returns the equivalence class for the provided list of probabilities
            result = classibu.svmPredictProbs(newProbs[i], equiv)
            modelDF = defaultdict(lambda: {})

            actualClass = probaDF.iloc[i]["Actual Class"]

            modelDF["Equiv Class"] = equiv.name
            modelDF["Actual Class"] = actualClass
            modelDF["Class Size"] = len(result)
            modelDF["Correct Class Predicted"] = actualClass in result
            modelDF["Exact User Predicted"] = actualClass == result[0]
            modelDF["Largest Possible Class Size"] = len(probaDF["Actual Class"])
            modelDF["Predicted Class"] = result[0]

            modelParams = probaDF.drop(list(probaDF.filter(regex=r"Class|\d|User")), axis=1)

            for paramName, paramValue in modelParams.items():
                modelDF[paramName] = paramValue[1]

            equivResults.append(dict(modelDF))
    return equivResults

"""
Formats the probability DataFrame to allow for use for predictions

Ensures that the probabilities are provided as a list.
This list will be in ascending order of userIDs.

This should return a list of a list of probabilities, the super list being in order of
the predicted test instances, which it already is.
"""
def formatProbs(probaDF):
    probs = []

    probaDF = probaDF.rename(columns=lambda x: re.sub("User ID ", "", x))

    # Filter the DF to only get all the probabilities
    filtCols = list(probaDF.filter(regex=r"\d").columns)
    userCols = pd.concat([probaDF.pop(myCol) for myCol in filtCols], 1)
    
    userCols = userCols.rename(columns=lambda x: int(x))
    userCols = userCols.reindex(np.sort(userCols.columns), axis=1)

    # Store each row of probabilities into the super list of probabilities
    for _, probSet in userCols.iterrows():
        probs.append(list(probSet))

    # Forming the new reformatted DF
    for myCol in userCols.columns:
        probaDF[myCol] = userCols.pop(myCol).values

    # Return both the list of probabilities and the formatted df
    return probs, probaDF

def plotEquivs(myArgs):
    # Set theme
    sns.set_style('whitegrid')
    
    acciesDF = None

    if myArgs.full:
        acciesDF = pd.read_csv(f"classySVM_FullSearchAccResultsFormatted.csv")
        # acciesDF = pd.read_csv(f"classySVM_FullSearchAccResults.csv")
    else:
        acciesDF = pd.read_csv(f"classySVM_{myArgs.searchNum}SearchAccResults.csv")

    acciesDF.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1, inplace=True)

    pprint(acciesDF[acciesDF["Equivalence Method"] == "JUMP"])

    #Plot the relationship between class accuracy and class size
    plotti = sns.scatterplot(x="kernel", y="Class Accuracy", hue="Equivalence Method", data=acciesDF) \
                .set_title("Relation between class accuracy and SVM Kernel")
    # plotti = sns.violinplot(x="Equivalence Method", y="Class Accuracy", data=acciesDF) \
        # .set_title("Comparision between equivalence class methods")
    # plotti = sns.violinplot(x="" data=acciesDF)
    # plotti = sns.heatmap(acciesDF.corr()).set_title("Correlation between class accuracy and user accuracy")

    if myArgs.saveAs:
        plotti.get_figure().savefig(myArgs.saveAs)

    plt.show()

def main(myArgs):
    cvResults = None

    if myArgs.full:
        cvResults = fileIO.loadPickle("classySVM_FullSearch.pkl").cv_results_
    else:
        cvResults = fileIO.loadPickle(f"classySVM_{myArgs.searchNum}Searches.pkl").cv_results_

    if myArgs.probsOnly:
        accResults = probWrapper(cvResults, myArgs.mini, single=True)
    else:
        #Stores the results for the accuracy tests
        accResults = testWrapper(fileName=myArgs.fileName)

        #Saving the results of the accuracy tests to a CSV
        if myArgs.full:
            accResults.to_csv(f"classySVM_FullSearchAccResults.csv")
        else:
            accResults.to_csv(f"classySVM_{myArgs.searchNum}SearchAccResults.csv")


if __name__ == "__main__":
    myParser = ArgumentParser()

    myParser.add_argument("--full", default=False, action="store_true",
                            help="Whether to use the full search data. Default: False")
    myParser.add_argument("--searchNum", "--num", default=10, metavar="NUM", type=int,
                            help="The number of hyper-parameter possibilities to search for.")
    myParser.add_argument("--plot", default=False, action="store_true",
                            help="Whether to plot a graph of collected results.")
    myParser.add_argument("--saveAs", default="myPlot.png", 
                            help="File name to save the plot as.")
    myParser.add_argument("--mini", "-m", default=False, action="store_true",
                        help="Whether a small subsection of the dataset will be used for training."
                            + "Default: False")
    myParser.add_argument("--fileName", default="AccuracyTests.csv",
                        help="The file name to store the results as.")
    myParser.add_argument("--complete", default=True, action="store_false",
                        help="Flag to turn off using a complete-incremental approach")
    myParser.add_argument("--runProbs", default=False, action="store_true",
                        help="Generate the probabilities from scratch when experimenting.")
    myParser.add_argument("--probsOnly", default=False, action="store_true",
                        help="Only save the probabilities and don't run the experimentation.")
    # I had this for debug things, but it's not appropriate in the final report
    # myParser.add_argument("--debug", default=False, action="store_true",
    #                     help="Legit. What do you think this does?")

    myArgs = myParser.parse_args()

    if myArgs.plot:
        plotEquivs(myArgs)
    else:
        main(myArgs)