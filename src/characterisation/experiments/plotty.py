import os
import re

#pylint: disable=import-error
import pandas as pd
import seaborn as sns

from argparse import ArgumentParser
from collections import defaultdict
from enum import Enum
from pprint import pprint

from matplotlib import pyplot as plt

ChartType = Enum("ChartTypes", "BAR LINE")

filePath = os.path.dirname(os.path.realpath(__file__))
dataPath = os.path.join(filePath, "..", "..", "..", "expResults")

"""
Return a sutiably formatted string pertaining to the supplied feature subset.
"""
def formatSubsets(subset):
    if subset == "Stop_Rich_AvgWords":
        return "Complete"
    else:
        subset = re.sub("Words", "", subset)
        subset = re.sub("Stoppies", "Stop-Words", subset)
        subset = re.sub("Stop_", "Stop-Words & ", subset)
        subset = re.sub("Rich$", "Richness", subset)
        subset = re.sub("Rich_", "Richness & ", subset)
        subset = re.sub("Avg", "Averages", subset)

        return subset

# * I need to determine what kind of plots I need
# * I need a standard bar graph plotting the accuracies for each equiv class
# * One bar graph for each feature set group and one for each hyper-parameter distribution
# ! I should have one for equiv class sizes
# * This would plot the relation between equiv class size and class prediction accuracy
# ? Maybe I could have a plot for the distribution of user classes being predicted
# * So just say that user 10364 is predicted 528 times etc. or something
# ! I should also plot the user predicted accuracy for each of the feature sets
# * They'll be compared to one another as well.
# ! Plot precision and recall

"""
Calculate the accuracy for each of the equivalence classes. Additionally calculate the
accuracy for each model's ability to predict individual users.

Return a dataframe containing the accuracies of each model parameter distribution and
for each equivalence class it has.
"""
def calcAccies(resultsDF):
    # This is ugly, I know, but I want to make sure it's stored bas
    # acciesDF = defaultdict(lambda: defaultdict(lambda: None))
    acciesDF = []

    equivs = set(resultsDF["Equiv Class"].values)

    # Looks like I don't need to worry about forming datasets with each hyperparameter
    # combination. Looks like the results are formed from the best performing models
    # from the hyperparameter search.
    for equiv in equivs:
        equivAccies = defaultdict(lambda: 0)
        iterDF = resultsDF[resultsDF["Equiv Class"] == equiv]
        numRecords = iterDF.shape[0]

        for _, myRow in iterDF.iterrows():
            # ? I might want a count of how many times each class size is encountered
            # ? so not just how many times they are correct.

            if myRow["Exact User Predicted"]:
                equivAccies["Correct User Predictions"] += 1
            if myRow["Correct Class Predicted"]:
                equivAccies["Correct Class Predictions"] += 1

        equivAccies["Equiv Class"] = equiv
        equivAccies["User Accuracy"] = 100 * equivAccies["Correct User Predictions"] / numRecords
        equivAccies["Equiv Class Accuracy"] = 100 * equivAccies["Correct Class Predictions"] / numRecords
        equivAccies["User Accuracy String"] = f"{100 * equivAccies['Correct User Predictions'] / numRecords:.5f}%"
        equivAccies["Equiv Class Accuracy String"] = f"{100 * equivAccies['Correct Class Predictions'] / numRecords:.5f}%"
        equivAccies["Record Count"] = numRecords

        equivAccies = dict(equivAccies)
        acciesDF.append(equivAccies)
        print(f"{equivAccies['Equiv Class']}: {equivAccies['Equiv Class Accuracy String']}")
        # print(equivAccies["Equiv Class Accuracy String"])

    print(f"User Accuracy: {acciesDF[0]['User Accuracy String']}")
    acciesDF = pd.DataFrame(acciesDF)
    return acciesDF

"""
Automatically plots all of the graphs as per the accuracies of the
generated probabilities.
"""
def plot():
    sns.set_style('whitegrid')
    allDFs = []

    resultDirs = list(os.walk(dataPath))[0][1]

    for myDir in resultDirs:
        dirPath = os.path.join(dataPath, myDir)
        print(formatSubsets(myDir))
        print(dirPath)

        resultsDF = pd.read_csv(os.path.join(dirPath, "Results.csv"))
        resultsDF.drop(list(resultsDF.filter(regex="Unnamed*")), axis=1, inplace=True)
        acciesDF = calcAccies(resultsDF)
        acciesDF["Feature Set"] = myDir
        acciesDF.fillna(0, inplace=True, axis=1)

        if myDir == "AveragesWords":
            print(f"DF Cols: {resultsDF.columns}")

        plotAccies(acciesDF, myDir)
        plotEquivs(resultsDF, myDir)

        allDFs.append(acciesDF)
        print()

    completeDF = pd.DataFrame().append(allDFs)
    completeDF.fillna(0, inplace=True)

    plotCorr(completeDF)

"""
Plot the graphs relating to the accuracy of equivalence class predictions 
and user predictions.
"""
def plotAccies(acciesDF, featureSubset, chartType=ChartType.BAR):
    plotTitle = f"Equivalence Class Accuracy for the {formatSubsets(featureSubset)} Subset of Features"
    plotti = None

    plt.clf() # ! Clearing plots so they don't overlap

    if chartType == ChartType.BAR:
        plotti = sns.barplot(x="Equiv Class", y="Equiv Class Accuracy", data=acciesDF)\
            .set_title(plotTitle)
    plotti.get_figure().savefig(os.path.join(dataPath, featureSubset, "Accuracy Chart.png"))

"""
Determines the correlation between the accuracies for each of the equivalence class
algorithms and the user accuracy. The result is subsequently plotted. 
"""
def plotCorr(completeDF):
    filtAccies = completeDF.filter(regex="Accuracy$|Equiv Class$")
    equivs = set(filtAccies["Equiv Class"].values)

    # ? Makes a seaborn correlation plot for each equiv class
    for equiv in equivs:
        plt.clf() # ! Clearing plots so they don't overlap
        equivSubset = filtAccies[filtAccies["Equiv Class"] == equiv]

        plotti = sns.heatmap(equivSubset.drop("Equiv Class", axis=1).corr())\
            .set_title(f"Correlation for the {equiv} equivalence class")
        plotti.get_figure().savefig(os.path.join(dataPath, f"{equiv}_Correlation.png"))

"""
Given the dataframe made from the user and equivalence class accuracies for a single model, 
make a new dataframe focused on equivalence classes. This new DF should allow for showing the 
relationship between attributes like class size to accuracy.
"""
def plotEquivs(resultsDF, featureSubset):
    # * Sum up the correct counts for each class size and calculate the accuracy.
    plotTitle = f"Equivalence Class Size-Accuracy Relationship ({formatSubsets(featureSubset)} Subset)"
    allAccies = []
    sizesDF = []

    plt.clf() # ! Clearing plots so they don't overlap

    equivs = set(resultsDF["Equiv Class"].values)
    for equiv in equivs:
        classAccies = defaultdict(lambda: 0)
        iterDF = resultsDF[resultsDF["Equiv Class"] == equiv]

        print(f"PlotEquivs - Iter DF Shape: {iterDF.shape}")

        for _, myRow in iterDF.iterrows():
            if myRow["Correct Class Predicted"]:
                classAccies[f"Correct Class Size {myRow['Class Size']} Predictions"] += 1
            else:
                classAccies[f"Incorrect Class Size {myRow['Class Size']} Predictions"] += 1

        classAccies["Equiv Class"] = equiv

        # Getting all the unique class sizes
        classSizes = set([int(keySplit) for myKey in classAccies.keys()\
                            for keySplit in myKey.split() if keySplit.isdigit()])

        for size in classSizes:
            sizeDict = defaultdict(lambda: 0)

            correctRegex = re.compile(f"Correct Class Size {size} Predictions")
            incorrectRegex = re.compile(f"Incorrect Class Size {size} Predictions")

            # Making sure both entries are initialised
            classAccies[f"Correct Class Size {size} Predictions"]\
                = classAccies[f"Correct Class Size {size} Predictions"]
            classAccies[f"Incorrect Class Size {size} Predictions"]\
                = classAccies[f"Incorrect Class Size {size} Predictions"]
            
            numCorrect = classAccies[list(filter(correctRegex.match, classAccies.keys()))[0]]
            numIncorrect = classAccies[list(filter(incorrectRegex.match, classAccies.keys()))[0]]

            classAccies[f"Class Size {size} Accuracy"] = 100 * numCorrect / (numCorrect + numIncorrect)
            
            sizeDict["Equiv Class"] = equiv
            sizeDict["Class Size"] = size
            sizeDict["Class Count"] = numCorrect + numIncorrect
            sizeDict["Accuracy"] = 100 * numCorrect / (numCorrect + numIncorrect)

            sizesDF.append(dict(sizeDict))
        allAccies.append(dict(classAccies))
    
    sizesDF = pd.DataFrame(sizesDF)
    
    if featureSubset == "Stop_Rich_AvgWords":
        print(sizesDF)

    # plotti = sns.scatterplot(x="Class Size", y="Accuracy", hue="Equiv Class", data=sizesDF)\
    #   .set_title(plotTitle)
    # plotti.savefig(os.path.join(dataPath, featureSubset, "SizeAccuracy.png"))
    
    # ? Plotting a line of best fit on a scatter graph
    # //plotti = sns.lmplot(x="Class Size", y="Accuracy", data=sizesDF)
    
    # //plt.title(plotTitle)
    # //plt.xlim(0, None)
    # //plt.ylim(0, None)

    # //myFig = plt.gcf()
    # //myFig.set_size_inches(10, 7.5)

    # //myFig.savefig(os.path.join(dataPath, featureSubset, "SizeAccuracyLOBF.png"))

    plotti = sns.lineplot(x="Class Size", y="Class Count", hue="Equiv Class", data=sizesDF)\
        .set_title("Class Size Occurrences")
    plotti.get_figure().savefig(os.path.join(dataPath, featureSubset, "SizeOccurrences.png"))

if __name__ == "__main__":
    plot()