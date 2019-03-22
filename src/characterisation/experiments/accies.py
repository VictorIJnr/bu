#Experiment file testing against the accuracies of different models 
import os

import seaborn as sns
import pandas as pd
import numpy as np

from argparse import ArgumentParser
from collections import Counter, defaultdict
from pprint import pprint

from matplotlib import pyplot as plt

from characterisation.classibu import skippedSVM, svmPredict, rawSVMPredict
from characterisation.classification.equiv import Equivs 
from characterisation.classification.sklearnSVM import expPredict, initSVM 
from characterisation.classification.sklearnHelper import split, filteredMap

from helpers import fileIO

"""
Wraps the functionality of performing accuracy tests into one method dependant
on having Cross-Validation results.
"""
def testWrapper(cvResults, mini=False, splitData=None, fileName="", complete=True, single=False):
    resultsDF = formatCVResults(cvResults)

    return runAccuracyTests(resultsDF, mini, splitData, fileName=fileName,
                            complete=complete, single=single)

def probWrapper(cvResults, mini=False, splitData=None, fileName="", complete=True, single=False):
    resultsDF = formatCVResults(cvResults)

    return runPredictions(resultsDF, mini, splitData, fileName=fileName,
                            complete=complete, single=single)


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
I also need to save files based on the subset of features being used for training
"""
def runPredictions(myDF, mini=False, splitData=None, fileName="", complete=True, single=False):
    preds = []
    cleanupFiles = []

    loopIters = 0
    for _, model in myDF.iterrows(): 
        if loopIters == 1 and single:
            break

        intermediateFile = fileName[:-4] + model["params"]["kernel"] + ".csv"

        # Prediction files exist for each model
        # Each file contains the predicted probabilities for each of the user classes
        # Each record is the probabilities for each test instance
        # Also contain predicted class and actual class,
        # Things such as equiv class sizes can be determined from the probabilities

        # Here we save each individual prediction file,
        # Prediction file uses the non-experimental methods and saves all of the
        # Predictions i

        preds.extend(predictTests(model, splitData, mini, fileName))

        pd.DataFrame(preds).to_csv(intermediateFile)
        
        #This could probably be determined from outside the loop
        #I mean the array of possible values for the kernel parameter
        # kernelValues.append(model[])
        cleanupFiles.append(intermediateFile)
        
        loopIters += 1

    for cleanFile in cleanupFiles:
        os.remove(cleanFile)

    preds = pd.DataFrame(preds)
    preds.to_csv(fileName)

    return pd.DataFrame(preds)

"""
Calculates the probabilities for each of the trained user classes for each 
test instance (either provided or generated automatically)
"""
def predictTests(model, splitData, mini, fileName):
    predResults = []

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
    
    # Don't predict against the equiv classes here
    # Instead just get the raw classification probabilities
    # We'll use yTest to get the actual class obviously
    # All of the trained user classes, just to associate their probabilities

    for i in np.arange(len(xTest)):
        predicty = defaultdict(lambda: None)
        userIDs = list(filteredMap().keys())

        testInput = xTest[i]
        actualClass = yTest[i]
        
        predProbs = rawSVMPredict(myModel, testInput)
        predClass = userIDs[np.argmax(predProbs)]
        
        predicty.update({f"User ID {userID}": prob for userID, prob in zip(userIDs, predProbs)})

        pprint(dict(predicty))

        pprint(xTest)
        pprint(yTest)

        predicty["Actual Class"] = actualClass
        predicty["Predicted Class"] = predClass
        predicty["User Predicted"] = actualClass == predClass

        predResults.append(predicty)


    return predResults

"""
Runs tests against the accuracy of models

@param single indicates whether to test only the best performing model,
    determined by the Cross-Validation
"""
def runAccuracyTests(myDF, mini=False, splitData=None, fileName="", complete=True, single=False):
    accResults = []
    # kernelValues = []
    cleanupFiles = []

    loopIters = 0
    for _, model in myDF.iterrows(): 
        if loopIters == 1 and single:
            break

        intermediateFile = fileName[:-4] + model["params"]["kernel"] + ".csv"

        accResults.extend(modelAccTest(model, mini=mini, splitData=splitData, 
                                        fileName=fileName, complete=complete))

        pd.DataFrame(accResults).fillna(0).to_csv(intermediateFile)
        
        #This could probably be determined from outside the loop
        #I mean the array of possible values for the kernel parameter
        # kernelValues.append(model[])
        cleanupFiles.append(intermediateFile)
        
        loopIters += 1

    for file in cleanupFiles:
        os.remove(file)

    accResults = pd.DataFrame(accResults).fillna(0)
    accResults.to_csv(fileName)

    return pd.DataFrame(accResults).fillna(0)

"""
Tests a single model instance from the CV Results
"""
def modelAccTest(model, mini=False, splitData=None, fileName="", complete=True):
    accResults = []

    print("\nCurrently evaluating this model:")
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
    
    if complete:
        accResults.extend(incrementalEquivsTests(myModel, xTest, yTest, 
                            model["params"], fileName=fileName))
    else:
        accResults.extend(testEquivs(myModel, xTest, yTest, 
                            model["params"], fileName=fileName))

    pd.DataFrame(accResults).to_csv(fileName)
    return accResults

"""
Testing against all of the different equivalence classes
"""
def testEquivs(myModel, xTest, yTest, modelParams=None, fileName=""):
    equivResults = []

    for equiv in Equivs:
        print(f"Testing the {equiv.name} equivalence class.")
        modelResults = defaultdict(lambda: {})
        claccuracy, indAccuracy, equivClasses = expPredict(myModel, xTest, yTest, equivClass=equiv, returnProbs=True)
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

        pd.DataFrame(equivResults).to_csv(fileName)
    
    return equivResults

"""
Runs all of the accuracy tests incrementally such that all of the required CSVs are generated 
line-by-line,
Also synonymous to a "complete" CSV generation, checking against individual test classes
"""
def incrementalEquivsTests(myModel, xTest, yTest, modelParams=None, fileName=""):
    equivResults = []

    for i in np.arange(xTest.shape[0]):
        print(f"Testing index {i + 1} of "
               + f"{xTest.shape[0]} - {((i + 1)/ xTest.shape[0]) * 100:.3f}% done.",
                end="\r")
        
        for equiv in Equivs:
            result = svmPredict(myModel, xTest[i], equiv)
            modelDF = defaultdict(lambda: {})
            
            modelDF["Equiv Class"] = equiv.name
            modelDF["Actual Class"] = yTest[i]
            modelDF["Class Size"] = len(result)
            modelDF["Correct Class Predicted"] = yTest[i] in result
            modelDF["Exact User Predicted"] = yTest[i] == result[0]
            modelDF["Largest Possible Class Size"] = len(np.unique(yTest))
            modelDF["Predicted Class"] = result[0]

            if modelParams is not None:
                for paramName, paramValue in modelParams.items():
                    modelDF[paramName] = paramValue

            equivResults.append(dict(modelDF))

            pd.DataFrame(equivResults).to_csv(fileName)
    
    return equivResults

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
        accResults = probWrapper(cvResults, myArgs.mini, 
                                    fileName=myArgs.fileName, complete=myArgs.complete, single=True)
    else:
        #Stores the results for the accuracy tests
        accResults = testWrapper(cvResults, myArgs.mini, 
                                    fileName=myArgs.fileName, complete=myArgs.complete)

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
    myParser.add_argument("--saveProbs", default=True, action="store_false",
                        help="Turn off the saving of probabilities when experimenting.")
    myParser.add_argument("--probsOnly", default=False, action="store_true",
                        help="Only save the probabilities and don't run the experimentation.")

    myArgs = myParser.parse_args()

    if myArgs.plot:
        plotEquivs(myArgs)
    else:
        main(myArgs)