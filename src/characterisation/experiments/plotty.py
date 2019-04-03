
#pylint: disable=import-error
import pandas as pd
import seaborn as sns

from argparse import ArgumentParser

def plot(myArgs):
    pass

if __name__ == "__main__":
    myParser = ArgumentParser()

    myParser.add_argument("--data", help="The data file to plot into a graph.")

    myArgs = myParser.parse_args()

    plot(myArgs)