#Experiment file testing against the accuracies of different models 

import seaborn as sns
import pandas as pd
import numpy as np

from argparse import ArgumentParser
from collections import Counter, defaultdict
from pprint import pprint

from matplotlib import pyplot as plt

from characterisation.classibu import skippedSVM
from characterisation.classification.equiv import Equivs 
from characterisation.classification.sklearnSVM import expPredict, initSVM 
from characterisation.classification.sklearnHelper import split

from helpers import fileIO

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
Runs tests against the accuracy of models
"""
def runAccuracyTests(myDF, mini=False, dataDF=None):
    accResults = []

    for _, model in myDF.iterrows(): 
        print("\nCurrently evaluating this model:")
        pprint(model)

        myModel = None
        paramDist = {paramName: [paramValue] for paramName, paramValue in model["params"].items()}

        xTest, yTest = None, None

        if dataDF is None:
            myModel, xTest, yTest = skippedSVM(paramDist=paramDist, mini=mini, searchNum=1,
                                        returnTest=True, verbose=False)
        else:
            xTrain, yTrain, xTest, yTest = split(myDF=dataDF)
            myModel = initSVM(xTrain, yTrain, paramDist=paramDist, fullSearch=True)
        
        for equivTest in testEquivs(myModel, xTest, yTest, model["params"]):
            accResults.append(equivTest)
    
    accResults = pd.DataFrame(accResults)
    return accResults.fillna(0)

def testEquivs(myModel, xTest, yTest, modelParams=None):
    equivResults = []

    for equiv in Equivs:
        print(f"Testing the {equiv.name} equivalence class.")
        modelResults = defaultdict(lambda: {})
        claccuracy, indAccuracy, equivClasses = expPredict(myModel, xTest, yTest, equivClass=equiv)

        #Loop over the equivalence class sizes I guess
        equivSizes = [len(equivClass) for equivClass in equivClasses]
        equivSizeCounter = Counter(equivSizes)

        modelResults["Equivalence Method"] = equiv.name
        modelResults["User Accuracy"] = indAccuracy
        modelResults["Class Accuracy"] = claccuracy
        modelResults["Min Class Size"] = np.amin(equivSizes)
        modelResults["Max Class Size"] = np.amax(equivSizes)
        modelResults["Mean Class Size"] = np.mean(equivSizes)
        modelResults["Largest Possible Class Size"] = len(np.unique(yTest))

        for i in np.arange(1, np.amax(equivSizes) + 1):
            modelResults[f"Class Size {i} Count"] = equivSizeCounter[i]

        if modelParams is not None:
            for paramName, paramValue in modelParams.items():
                modelResults[paramName] = paramValue

        equivResults.append(dict(modelResults))
    
    return equivResults

"""
Wraps the functionality of performing accuracy tests into one method dependant
on having Cross-Validation results.
"""
def testWrapper(cvResults, mini=False, dataDF=None):
    resultsDF = formatCVResults(cvResults)

    return runAccuracyTests(resultsDF, mini, dataDF)

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

    #Stores the results for the accuracy tests
    accResults = testWrapper(cvResults, myArgs.mini)

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

    myArgs = myParser.parse_args()

    if myArgs.plot:
        plotEquivs(myArgs)
    else:
        main(myArgs)