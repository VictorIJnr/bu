# Running experiments (kinda) for SVM hyperparameter search

from argparse import ArgumentParser
from pprint import pprint

from helpers import fileIO

import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

from characterisation.classibu import reducedSVM, skippedSVM

def main():
    pass

def train(myArgs):
    mySVM = None
    if myArgs.skip:
        mySVM = skippedSVM(mini=myArgs.mini, folds=myArgs.folds, searchNum=myArgs.searchNum, 
            full=myArgs.fullSearch)
    else:
        mySVM = reducedSVM(mini=myArgs.mini, folds=myArgs.folds, searchNum=myArgs.searchNum, 
            full=myArgs.fullSearch)
        

def graphy(myArgs):
    hyperCV = None
    if myArgs.fullSearch:
        hyperCV = fileIO.loadPickle("classySVM_FullSearch.pkl").cv_results_
    else:
        hyperCV = fileIO.loadPickle(f"classySVM_{myArgs.searchNum}Searches.pkl").cv_results_

    hyperCV["avgTestScore"] = hyperCV["split0_test_score"] + hyperCV["split1_test_score"] \
                                + hyperCV["split2_test_score"] + hyperCV["split3_test_score"] \
                                + hyperCV["split4_test_score"] 
    hyperCV["avgTestScore"] /= 5 

    hyperCV["avgTrainScore"] = hyperCV["split0_train_score"] + hyperCV["split1_train_score"] \
                                + hyperCV["split2_train_score"] + hyperCV["split3_train_score"] \
                                + hyperCV["split4_train_score"] 
    hyperCV["avgTrainScore"] /= 5 

    del hyperCV["mean_fit_time"]
    del hyperCV["std_fit_time"]
    del hyperCV["mean_score_time"]
    del hyperCV["std_score_time"]
    del hyperCV["split0_test_score"]
    del hyperCV["split1_test_score"]
    del hyperCV["split2_test_score"]
    del hyperCV["split3_test_score"]
    del hyperCV["split4_test_score"]
    del hyperCV["split0_train_score"]
    del hyperCV["split1_train_score"]
    del hyperCV["split2_train_score"]
    del hyperCV["split3_train_score"]
    del hyperCV["split4_train_score"]
    del hyperCV["params"]

    print(hyperCV.keys())
    hyperDF = pd.DataFrame(hyperCV)

    pprint(hyperDF)

    # liney = sns.lineplot(x="param_coef0", y="avgTestScore", hue="param_kernel", data=hyperDF)
    # liney = sns.scatterplot(x="param_coef0", y="avgTestScore", hue="param_kernel", data=hyperDF)
    liney = sns.heatmap(hyperDF["avgTestScore"].corr(hyperDF["avgTrainScore"]))
    # liney = sns.heatmap(hyperDF.corr())
    liney = liney.get_figure()

    liney.savefig("Heatmappy.png")
    # liney.savefig("CoefKernels.png")
    plt.show()

    if myArgs.fullSearch:
        hyperDF.to_csv(f"classySVM_FullSearchResults.csv")
    else:
        hyperDF.to_csv(f"classySVM_{myArgs.searchNum}SearchesResults.csv")

if __name__ == "__main__":
    myParser = ArgumentParser()

    myParser.add_argument("--train", default=False, action="store_true",
                        help="Whether to solely train models according to a "
                            + "hyper-parameter search.")
    myParser.add_argument("--searchNum", "--num", default=10, metavar="NUM", type=int,
                        help="The number of hyper-parameter possibilities to search for.")
    myParser.add_argument("--fullSearch", default=False, action="store_true",
                        help="Whether to search the complete hyper-parameter space.")
    myParser.add_argument("--mini", "-m", default=False, action="store_true",
                        help="Whether a small subsection of the dataset will be used for training."
                            + "Default: False")
    myParser.add_argument("--skip", default=False, action="store_true",
                        help="Whether to skip dimension reduction with the convolutional autoencoder"
                            + "Default: False")
    myParser.add_argument("--folds", default=5, type=int,
                        help="The number of folds used for cross-fold validation.")
    myParser.add_argument("--verbose", default=False, action="store_true")

    myArgs = myParser.parse_args()

    if myArgs.train:
        train(myArgs)
    else:
        graphy(myArgs)