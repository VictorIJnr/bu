import os
import re
import sys

import numpy as np
import pandas as pd
import spacy as sp

from collections import Counter, defaultdict
from functools import reduce
from pprint import pprint

worldbuilding = "worldbuilding.stackexchange.com"
serverfault = "serverfault.com"

spacy = sp.load("en")

dataset = worldbuilding
buPath = os.path.dirname(os.path.realpath(__file__))
dataPath = os.path.join(buPath, "..", "..", "data")

#Features
"""
Number of words / characters in a post
Vocabulary Richness:    
    Yule's K,
    Number of Hapax/Dis/Tris/Tetrakis Legomena
"""

"""
Returns a dictionary containing the number of times a word appeared in
the corpus n times.
E.g. given 3 words appearing 5 times, the frequency 5, will be given a value
of 3.
"""
def metaFrequencies(wordCounts):
    return Counter(wordCounts.values())

"""
Gets the number of words in the corpus
"""
def totalNumWords(doc):
    #Splitting each word, previously token based, which lead to extra words
    return sum([len(str(sentence).split()) for sentence in doc.sents])

"""
Gets the number of characters within a corpus
"""
def totalNumChars(doc):
    words = [str(sentence).split() for sentence in doc.sents]
    
    #Converting "words" from a 2D array of words (per sentence) into a 1D array of all words
    words = reduce(lambda x, y: x + y, words)
    return sum([len(word) for word in words])

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

"""
Determines the average number of words in a sentence
"""
def avgSentenceWords(doc):
    numSentences = sum(1 for _ in doc.sents) #Counting the number of sentences in the text
    return totalNumWords(doc) / numSentences

"""
Determines the average number of characters in a sentence
"""
def avgSentenceChars(doc):
    numSentences = sum(1 for _ in doc.sents) #Counting the number of sentences in the text
    return totalNumChars(doc) / numSentences

"""
Returns the frequency of all the stop words
"""
def stopWordFreq(wordCounts, stopWords):
    stopCounter = defaultdict(lambda: -1)

    #Add all stop words to the counter, after which, all their counts will be 0
    for word in stopWords:
        stopCounter[word] += 1

    for (key, value) in wordCounts.items():
        if key in stopWords:
            stopCounter[key] = value

    return stopCounter

def loadStopWords():
    return [line.rstrip("\n") for line in open(os.path.join(buPath, "stopwords.txt"))]

def main():
    postDF = pd.read_csv(os.path.join(dataPath, dataset, "RestrictedPosts.csv"))
    body = re.sub(re.compile("<.*?>|\r?\n|\r"), "", postDF.iloc[1]["Body"])
    doc = spacy(body)
    stopWords = loadStopWords()
    
    words = [token.text for token in doc if token.is_punct == False]
    wordCounts = Counter(words)
    print(f"Total number of words:\t{totalNumWords(doc)}")
    print(f"Total number of characters:\t{totalNumChars(doc)}")
    print(f"Average Number of Words in a Sentence:\t{avgSentenceWords(doc)}")
    print(f"Average Number of Characters in a Sentence:\t{avgSentenceChars(doc)}\n")
    [print(sentence) for sentence in doc.sents]

if __name__ == "__main__":
    main()