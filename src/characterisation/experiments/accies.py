#Experiment file testing against the accuracies of different models 

import pandas as pd

from argparse import ArgumentParser
from collections import defaultdict
from pprint import pprint

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

    accResults = []

    for _, model in hyperDF.iterrows(): 
        print("\nCurrently evaluating this model:")
        pprint(model)

        paramDist = {paramName: [paramValue] for paramName, paramValue in model["params"].items()}

        myModel, xTest, yTest = skippedSVM(paramDist=paramDist, searchNum=1, returnTest=True, verbose=False)
        
        for equiv in Equivs:
            modelResults = defaultdict(lambda: {})
            claccuracy, indAccuracy = expPredict(myModel, xTest, yTest, equivClass=equiv)

            modelResults["Equivalence Method"] = equiv.name
            modelResults["User Accuracy"] = indAccuracy
            modelResults["Class Accuracy"] = claccuracy

            for paramName, paramValue in model["params"].items():
                modelResults[paramName] = paramValue

            accResults.append(modelResults)
    
    accResults = pd.DataFrame(accResults)

    if myArgs.full:
        accResults.to_csv(f"classySVM_FullSearchAccResults.csv")
    else:
        accResults.to_csv(f"classySVM_{myArgs.searchNum}SearchAccResults.csv")

def plotAllEquivs(myModel):
    pass

if __name__ == "__main__":
    myParser = ArgumentParser()

    myParser.add_argument("--full", default=False, action="store_true",
                            help="Whether to use the full search data. Default: False")
    myParser.add_argument("--searchNum", "--num", default=10, metavar="NUM", type=int,
                            help="The number of hyper-parameter possibilities to search for.")

    myArgs = myParser.parse_args()

    main(myArgs)