import os

#pylint: disable=import-error
import pandas as pd
import seaborn as sns

from argparse import ArgumentParser
from enum import Enum
from pprint import pprint

from matplotlib import pyplot as plt

ChartType = Enum("ChartTypes", "BAR LINE")

filePath = os.path.dirname(os.path.realpath(__file__))
dataPath = os.path.join(filePath, "..", "..", "..", "expResults")

# * I need to determine what kind of plots I need
# * I need a standard bar graph plotting the accuracies for each equiv class
# * One bar graph for each feature set group and one for each hyper-parameter distribution
# ! I should have one for equiv class sizes
# * This would plot the relation between equiv class size and class prediction accuracy
# ? Maybe I could have a plot for the distribution of user classes being predicted
# * So just say that user 10364 is predicted 528 times etc. or something

"""
Calculate the accuracy for each of the equivalence classes. Additionally calculate the
accuracy for each model's ability to predict individual users.

Return a dataframe containing the accuracies of each model parameter distribution and
for each equivalence class it has.
"""
def calcAccies(resultsDF):
    acciesDF = pd.DataFrame()

    return acciesDF

def plot(myArgs):
    sns.set_style('whitegrid')

    resultDirs = list(os.walk(dataPath))[0][1]

    for myDir in resultDirs:
        dirPath = os.path.join(dataPath, myDir)
        print(dirPath)

        resultsDF = pd.read_csv(os.path.join(dirPath, "Results.csv"))
        acciesDF = calcAccies(resultsDF)

        plotAccies(acciesDF)

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
def plotAccies(acciesDF, chartType=ChartType.BAR):
    pass

if __name__ == "__main__":
    myParser = ArgumentParser()

    myParser.add_argument("--dataFile", help="The data file to plot into a graph.")

    myArgs = myParser.parse_args()

    plot(myArgs)