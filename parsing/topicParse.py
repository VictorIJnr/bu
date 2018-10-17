import json

import spacy as sp
import pandas as pd

from pprint import pprint

limit = 10
spacy = sp.lo   ad("en")

def parseJSON(jsonFile, parseLimit=None):
    with open(jsonFile) as training:
        data = json.load(training)
        
    dataLabels = list(data[0].keys())

    trainDF = pd.DataFrame.from_records(data[:parseLimit], columns=dataLabels)
    # pprint(data)
    # pprint(trainDF)
    return trainDF

def segmentation(input):
    print(f"{input}")
    segmented = spacy(f"{input}")
    for sentence in segmented.sents:
        pprint(sentence)
    print("\n")
    return segmented.sents

if __name__ == "__main__":
    jsonFile = "../data/TopicDialogues/train.json"
    df = parseJSON(jsonFile, limit)
    segementedThreads = []
    for i in range(len(df.iloc[5].get("thread"))):
        segmentation(df.iloc[5].get("thread")[i]["text"])
    pprint(segementedThreads)
