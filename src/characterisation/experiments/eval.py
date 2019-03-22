"""
Runs evaluation on any specified dataset.
"""

from argparse import ArgumentParser

from characterisation.classification.sklearnSVM import initSVM

def main():
    pass

def runPrediction():
    pass

if __name__ == "__main__":
    myParser = ArgumentParser()

    myParser.add_argument("--stoppies", default=False, action="store_true",
                        help="Whether to use the stop words section")
    myParser.add_argument("--richness", default=False, action="store_true",
                        help="Train the subset of features pertaining to vocabulary richness."
                            + "Default: False")
    myParser.add_argument("--averages", default=False, action="store_true",
                        help="Train the subset of features pertaining to averages in a user's corpus."
                            + "Default: False")
    myParser.add_argument("--mini", "-m", default=False, action="store_true",
                        help="Whether a small subsection of the dataset will be used for training."
                            + "Default: False")
    myParser.add_argument("--load", default=False, action="store_true",
                        help="Whether to load a previously trained model."
                            + "Default: False")
    main()