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
            # Just making sure it's initialised and doesn't get overriden
            equivAccies[f"Correct Class Size {myRow['Class Size']}"] = equivAccies[f"Correct Class Size {myRow['Class Size']}"]

            if myRow["Exact User Predicted"]:
                equivAccies["Correct User Predictions"] += 1
            if myRow["Correct Class Predicted"]:
                equivAccies["Correct Class Predictions"] += 1
                equivAccies[f"Correct Class Size {myRow['Class Size']}"] += 1

        equivAccies["Equiv Class"] = equiv
        equivAccies["User Accuracy"] = 100 * equivAccies["Correct User Predictions"] / numRecords
        equivAccies["Equiv Class Accuracy"] = 100 * equivAccies["Correct Class Predictions"] / numRecords
        equivAccies["User Accuracy String"] = f"{100 * equivAccies['Correct User Predictions'] / numRecords:.5f}%"
        equivAccies["Equiv Class Accuracy String"] = f"{100 * equivAccies['Correct Class Predictions'] / numRecords:.5f}%"

        equivAccies = dict(equivAccies)
        acciesDF.append(equivAccies)
        print(equivAccies)

    acciesDF = pd.DataFrame(acciesDF)
    return acciesDF

"""
Automatically plots all of the graphs as per the accuracies of the
generated probabilities.
"""
def plot(myArgs):
    sns.set_style('whitegrid')
    allDFs = []

    resultDirs = list(os.walk(dataPath))[0][1]

    for myDir in resultDirs:
        dirPath = os.path.join(dataPath, myDir)
        print(dirPath)

        resultsDF = pd.read_csv(os.path.join(dirPath, "Results.csv"))
        resultsDF.drop(list(resultsDF.filter(regex="Unnamed*")), axis=1, inplace=True)
        acciesDF = calcAccies(resultsDF)
        acciesDF["Feature Set"] = myDir

        print(formatSubsets(myDir))

        plotAccies(acciesDF, myDir)

        allDFs.append(acciesDF)

        #Get the path and join each of the files in the directory 

    # pprint(acciesDF[acciesDF["Equivalence Method"] == "JUMP"])

    # #Plot the relationship between class accuracy and class size
    # plotti = sns.scatterplot(x="kernel", y="Class Accuracy", hue="Equivalence Method", data=acciesDF) \
    #             .set_title("Relation between class accuracy and SVM Kernel")
    # # plotti = sns.violinplot(x="Equivalence Method", y="Class Accuracy", data=acciesDF) \
    #     # .set_title("Comparision between equivalence class methods")
    # # plotti = sns.violinplot(x="" data=acciesDF)
    # # plotti = sns.heatmap(acciesDF.corr()).set_title("Correlation between class accuracy and user accuracy")

    # plt.show()

"""
Plot the graphs relating to the accuracy of equivalence class predictions 
and user predictions.
"""
def plotAccies(acciesDF, featureSubset, chartType=ChartType.BAR):
    plotTitle = f"Equivalence Class Accuracy for the {formatSubsets(featureSubset)} Subset of Features"

    if chartType == ChartType.BAR:
        plotti = sns.barplot(x="Equiv Class", y="Equiv Class Accuracy", data=acciesDF)\
            .set_title(plotTitle)
    plotti.get_figure().savefig(os.path.join(dataPath, featureSubset, "Accuracy Chart.png"))

"""
Plot the graphs showing the relationship between equivalence class sizes and accuracy
"""
def plotSizes(sizesDF, featureSubset):
    plotTitle = f"Equivalence Class Size-Accuracy Relationship"

    

if __name__ == "__main__":
    myParser = ArgumentParser()

    myParser.add_argument("--dataFile", help="The data file to plot into a graph.")

    myArgs = myParser.parse_args()

    plot(myArgs)