# Running experiments (kinda) for SVM hyperparameter search

from argparse import ArgumentParser
from helpers import fileIO

import 
import seaborn as sns

from matplotlib import pyplot as plt

from characterisation.classibu import reducedSVM

def main():
    pass

def train(myArgs):
    reducedSVM(mini=myArgs.mini, folds=myArgs.folds, searchNum=myArgs.searchNum, 
        full=myArgs.fullSearch)

def graphy(myArgs):
    if myArgs.fullSearch:
        hyper = fileIO.loadPickle("classySVM_FullSearch.pkl")
    else:
        hyper = fileIO.loadPickle(f"classySVM_{myArgs.searchNum}Searches.pkl")

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
    myParser.add_argument("--folds", default=5, type=int,
                        help="The number of folds used for cross-fold validation.")
    myParser.add_argument("--verbose", default=False, action="store_true")

    myArgs = myParser.parse_args()

    if myArgs.train:
        train(myArgs)
    else:
        graphy()