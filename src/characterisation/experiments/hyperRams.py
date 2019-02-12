# Running experiments (kinda) for SVM hyperparameter search

from argparse import ArgumentParser

from characterisation.classibu import reducedSVM

def main():
    pass

def train(myArgs):
    reducedSVM(mini=myArgs.mini, folds=myArgs.folds, searchNum=myArgs.searchNum, 
        full=myArgs.fullSearch)

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

    myArgs = myParser.parse_args()

    if myArgs.train:
        train(myArgs)