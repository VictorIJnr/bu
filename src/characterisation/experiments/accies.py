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
from characterisation.classification.sklearnSVM import expPredict 

from helpers import fileIO

def main(myArgs):
    hyperDF = None
    if myArgs.full:
        hyperDF = fileIO.loadPickle("classySVM_FullSearch.pkl").cv_results_
    else:
        hyperDF = fileIO.loadPickle(f"classySVM_{myArgs.searchNum}Searches.pkl").cv_results_

    del hyperDF["mean_fit_time"]
    del hyperDF["std_fit_time"]
    del hyperDF["mean_score_time"]
    del hyperDF["std_score_time"]
    del hyperDF["split0_test_score"]
    del hyperDF["split1_test_score"]
    del hyperDF["split2_test_score"]
    del hyperDF["split3_test_score"]
    del hyperDF["split4_test_score"]
    del hyperDF["split0_train_score"]
    del hyperDF["split1_train_score"]
    del hyperDF["split2_train_score"]
    del hyperDF["split3_train_score"]
    del hyperDF["split4_train_score"]

    hyperDF = pd.DataFrame(hyperDF)
    hyperDF.sort_values("rank_test_score", inplace=True)

    pprint(hyperDF)

    accResults = []

    for _, model in hyperDF.iterrows(): 
        print("\nCurrently evaluating this model:")
        pprint(model)

        paramDist = {paramName: [paramValue] for paramName, paramValue in model["params"].items()}

        myModel, xTest, yTest = skippedSVM(paramDist=paramDist, mini=myArgs.mini, searchNum=1,
                                    returnTest=True, verbose=False)
        
        for equiv in Equivs:
            modelResults = defaultdict(lambda: {})
            claccuracy, indAccuracy, equivClasses = expPredict(myModel, xTest, yTest, equivClass=equiv)

            equivSizes = [len(equivClass) for equivClass in equivClasses]
            equivSizeCounter = Counter(equivSizes)

            pprint(equivSizeCounter)

            modelResults["Equivalence Method"] = equiv.name
            modelResults["User Accuracy"] = indAccuracy
            modelResults["Class Accuracy"] = claccuracy
            modelResults["Min Class Size"] = np.amin(equivSizes)
            modelResults["Max Class Size"] = np.amax(equivSizes)
            modelResults["Mean Class Size"] = np.mean(equivSizes)

            for i in np.arange(np.amax(equivSizes)):
                modelResults[f"Class Size {i} Count"] = equivSizeCounter[i]

            for paramName, paramValue in model["params"].items():
                modelResults[paramName] = paramValue

            accResults.append(modelResults)
    
    accResults = pd.DataFrame(accResults)

    if myArgs.full:
        accResults.to_csv(f"classySVM_FullSearchAccResults.csv")
    else:
        accResults.to_csv(f"classySVM_{myArgs.searchNum}SearchAccResults.csv")

def plotEquivs(myArgs):
    # Set theme
    sns.set_style('whitegrid')
    
    acciesDF = None

    if myArgs.full:
        acciesDF = pd.read_csv(f"classySVM_FullSearchAccResults.csv")
    else:
        acciesDF = pd.read_csv(f"classySVM_{myArgs.searchNum}SearchAccResults.csv")

    acciesDF.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1, inplace=True)

    pprint(acciesDF[acciesDF["Equivalence Method"] == "JUMP"])

    plotti = sns.scatterplot(x="kernel", y="Class Accuracy", hue="Equivalence Method", data=acciesDF) \
                .set_title("Relation between class accuracy and SVM Kernel")
    # plotti = sns.violinplot(x="Equivalence Method", y="Class Accuracy", data=acciesDF) \
        # .set_title("Comparision between equivalence class methods")
    # plotti = sns.violinplot(x="" data=acciesDF)
    # plotti = sns.heatmap(acciesDF.corr()).set_title("Correlation between class accuracy and user accuracy")

    if myArgs.saveAs:
        plotti.get_figure().savefig(myArgs.saveAs)

    plt.show()

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