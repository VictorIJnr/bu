    # Running experiments for the Convolutional AutoEncoder

from argparse import ArgumentParser
from collections import defaultdict
from pprint import pprint

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

from helpers import fileIO

from characterisation.helpers.reduction import initDatasetAE

def main():
    histories = []
    lossFuncs = ["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error",
                    "mean_squared_logarithmic_error", "squared_hinge", "hinge"]
    
    for lossy in lossFuncs:
        history = initDatasetAE(loss=lossy, returnHistory=True)[1].history

        for i in range(1, 11):
            lossResults = defaultdict(lambda: {})
            lossResults["epoch"] = i
            lossResults["loss_function"] = lossy
            lossResults["val_loss"] = history["val_loss"][i - 1]
            lossResults["val_acc"] = history["val_acc"][i - 1]
            lossResults["loss"] = history["loss"][i - 1]
            lossResults["acc"] = history["acc"][i - 1]

            histories.append(lossResults)

    fileIO.saveJSON(histories, "expConvAE.json")    

def makeGraphs():
    data = fileIO.loadJSON("expConvAE.json")

    myDF = pd.DataFrame(data)
    myDF = myDF[(myDF.loss_function != "mean_absolute_percentage_error")
                & (myDF.loss_function != "mean_squared_error")]
 
    liney = sns.lineplot(x="epoch", y="loss", hue="loss_function", data=myDF)
    liney = liney.get_figure()

    liney.savefig("MaxLossMAE.png")
    plt.show()

if __name__ == "__main__":
    myParser = ArgumentParser()

    myParser.add_argument("--graphs", default=False, action="store_true")

    myArgs = myParser.parse_args()

    if myArgs.graphs:
        makeGraphs()
    else:
        main()