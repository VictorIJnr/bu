import os
import re
import sys

import spacy as sp
import pandas as pd

from collections import Counter
from functools import reduce
from pprint import pprint

worldbuilding = "worldbuilding.stackexchange.com"
serverfault = "serverfault.com"

spacy = sp.load("en")

dataset = worldbuilding
buPath = os.path.dirname(os.path.realpath(__file__))
dataPath = os.path.join(buPath, "..", "..", "data")

commentDF = pd.read_csv(os.path.join(dataPath, dataset, "Comments.csv"))

"""
Returns a dictionary containing the number of times a word appeared in
the corpus n times.
E.g. given 3 words appearing 5 times, the frequency 5, will be given a value
of 3.
"""
def metaFrequencies(wordCounts):
    return Counter(wordCounts.values())

#Features
"""
Number of words / characters in a post
Vocabulary Richness:    
    Yule's K,
    Number of Hapax/Dis/Tris/Tetrakis Legomena
"""

#Yule's I
def yuleify(wordCounts):
    m1 = len(wordCounts)
    m2 = sum([key * value for (key, value) in metaFrequencies(wordCounts).items()])

    return m1 ** 2 / (m2 - m1)  if m2 - m1 != 0 else 0

"""
The number of words which occur 'n'(count) times within the given corpus
"""
def legomena(wordCounts, count=1):
    return metaFrequencies(wordCounts)[count]

def main():
    postDF = pd.read_csv(os.path.join(dataPath, dataset, "RestrictedPosts.csv"))
    body = re.sub(re.compile("<.*?>|\r?\n|\r"), "", postDF.iloc[0]["Body"])
    doc = spacy(body)

    words = [token.text for token in doc if token.is_punct == False]
    wordCounts = Counter(words)
    pprint(wordCounts)




if __name__ == "__main__":
    main()