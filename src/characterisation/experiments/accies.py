#Experiment file testing against the accuracies of different models 

import pandas as pd

from argparse import ArgumentParser
from collections import defaultdict
from pprint import pprint

from characterisation.classibu import reducedSVM
from characterisation.classification.equiv import Equivs 

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

    for model in hyperDF.iterrows(): 
        modelResults = defaultdict(lambda: {})
        print("\nThis model:")
        pprint(model)

        myModel = reducedSVM(paramDist=model["params"], searchNum=1)
        

def plotAllEquivs(myModel):
    pass

if __name__ == "__main__":
    myParser = ArgumentParser()

    myParser.add_argument("--full", default=False, action="store_true",
                            help="Whether to use the full search data. Default: False")

    myArgs = myParser.parse_args()

    main(myArgs)