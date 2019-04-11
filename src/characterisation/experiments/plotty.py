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
        equivAccies["Record Count"] = numRecords

        equivAccies = dict(equivAccies)
        acciesDF.append(equivAccies)
        print(f"{equivAccies['Equiv Class']}: {equivAccies['Equiv Class Accuracy String']}")
        # print(equivAccies["Equiv Class Accuracy String"])

    print(f"User Accuracy: {acciesDF[0]['User Accuracy String']}")
    acciesDF = pd.DataFrame(acciesDF)
    return acciesDF

"""
Given the dataframe made from the user and equivalence class accuracies for a single model, 
make a new dataframe focused on equivalence classes. This new DF should allow for showing the 
relationship between attributes like class size to accuracy.
"""
def calcEquivs(resultsDF):
    # * Sum up the correct counts for each class size and calculate the accuracy.

    print(completeDF.columns)

    filtSizesDF = completeDF.drop(list(completeDF.filter(regex="String|User")), axis=1)
    # filtSizesDF = completeDF.filter(regex="Correct Class Size (\d)+|Record Count")
    print(filtSizesDF.columns)
    print(filtSizesDF)

    # x-axis class size, y-axis accuracy

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
        print(formatSubsets(myDir))
        print(dirPath)

        resultsDF = pd.read_csv(os.path.join(dirPath, "Results.csv"))
        resultsDF.drop(list(resultsDF.filter(regex="Unnamed*")), axis=1, inplace=True)
        acciesDF = calcAccies(resultsDF)
        acciesDF["Feature Set"] = myDir
        acciesDF.fillna(0, inplace=True, axis=1)

        plotAccies(acciesDF, myDir)

        allDFs.append(acciesDF)
        print()


    completeDF = pd.DataFrame().append(allDFs)
    completeDF.fillna(0, inplace=True)

    calcEquivs(completeDF)
    plotCorr(completeDF)

    # print(completeDF)
    # print(completeDF.columns)

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
    plotti = None

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
        plt.clf()
        equivSubset = filtAccies[filtAccies["Equiv Class"] == equiv]

        plotti = sns.heatmap(equivSubset.drop("Equiv Class", axis=1).corr())\
            .set_title(f"Correlation for the {equiv} equivalence class")
        plotti.get_figure().savefig(os.path.join(dataPath, f"{equiv}_Correlation.png"))

"""
Plot the graphs showing the relationship between equivalence class sizes and accuracy
"""
def plotSizes(sizesDF, featureSubset, chartType=ChartType.BAR):
    plotTitle = f"Equivalence Class Size-Accuracy Relationship"
    plotti = None

    if chartType == ChartType.BAR:
        plotti = sns.barplot(x="Class Size", y="Equiv Class Accuracy", data=sizesDF)\
            .set_title(plotTitle)
    
    plt.show()


if __name__ == "__main__":
    myParser = ArgumentParser()

    myParser.add_argument("--dataFile", help="The data file to plot into a graph.")

    myArgs = myParser.parse_args()

    plot(myArgs)