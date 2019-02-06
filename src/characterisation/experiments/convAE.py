# Running experiments for the Convolutional AutoEncoder

from argparse import ArgumentParser
from collections import defaultdict

from helpers import fileIO

from characterisation.helpers.reduction import initDatasetAE

def main():
    histories = {}
    lossFuncs = ["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error",
                    "mean_squared_logarithmic_error", "squared_hinge", "hinge"]
    
    for lossy in lossFuncs:
        histories[lossy] = initDatasetAE(loss=lossy, returnHistory=True)[1].history

    print(histories)
    print(type(histories))

    fileIO.saveJSON(histories, "expConvAE.json")    

def makeGraphs():
    pass

if __name__ == "__main__":
    myParser = ArgumentParser()

    myParser.add_argument("--graphs", default=False, action="store_true")

    myArgs = myParser.parse_args()

    if myArgs.graphs:
        makeGraphs()
    else:
        main()