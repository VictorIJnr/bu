import json

import spacy as sp
import pandas as pd

from pprint import pprint

limit = 10
spacy = sp.lo   ad("en")

def parseJSON(jsonFile, parseLimit=None):
    with open(jsonFile) as training:
        data = json.load(training)
