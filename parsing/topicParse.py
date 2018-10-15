import json

import pandas as pd

from pprint import pprint

limit = 10

def parseJSON(jsonFile, parseLimit=None):
    with open(jsonFile) as training:
        data = json.load(training)
        
    dataLabels = list(data[0].keys())

    trainDF = pd.DataFrame.from_records(data[:parseLimit], columns=dataLabels)
    # pprint(data)
    pprint(trainDF)
    return trainDF

if __name__ == "__main__":
    jsonFile = "../data/TopicDialogues/train.json"
    parseJSON(jsonFile, limit)
