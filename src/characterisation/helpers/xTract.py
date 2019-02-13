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

debug = False

miniXtract = True  
singleXtract = False

dataset = worldbuilding
buPath = os.path.dirname(os.path.realpath(__file__))
dataPath = os.path.join(buPath, "..", "..", "..", "data")

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

"""
Yule's I
"""
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

def loadStopWords():
    return [line.rstrip("\n") for line in open(os.path.join(buPath, "stopwords.txt"))]

"""
Returns the frequency of all the stop words
"""
def stopWordFreq(wordCounts, stopWords=loadStopWords()):
    stopCounter = defaultdict(lambda: -1)

    #Add all stop words to the counter, after which, all their counts will be 0
    for word in stopWords:
        stopCounter[word] += 1

    for (key, value) in wordCounts.items():
        if key in stopWords:
            stopCounter[key] = value

    return stopCounter

def getWordCounts(doc):
    words = [token.text for token in doc if token.is_punct == False]
    return Counter(words)

"""
Chill out, this was just used for testing
"""
def main():
    postDF = pd.read_csv(os.path.join(dataPath, dataset, "RestrictedPosts.csv"))
    body = re.sub(re.compile("<.*?>|\r?\n|\r"), "", postDF.iloc[1]["Body"])
    doc = spacy(body)
    stopWords = loadStopWords()
    
    wordCounts = getWordCounts(doc)
    print(f"Total number of words:\t{totalNumWords(doc)}")
    print(f"Total number of characters:\t{totalNumChars(doc)}")
    print(f"Average Number of Words in a Sentence:\t{avgSentenceWords(doc)}")
    print(f"Average Number of Characters in a Sentence:\t{avgSentenceChars(doc)}\n")
    [print(sentence) for sentence in doc.sents]

"""
Runs feature extraction for a given CSV file relating to posts on StackExchange
Creates a new CSV file consisting of the extracted features
This is only for testing, execXtract is for general use
"""
def runExtraction(fileName):
    print(f"Debug xTract {os.path.join(dataset, fileName)}")
    execXtract(os.path.join(dataset, fileName))

def execXtract(fileName):
    dfCols = ["userID", "postID", "metaFreq", "numWords", "numChars", "yule",
                "hapaxLego", "disLego", "trisLego", "avgSentenceWords",
                "avgSentenceChars", "stopWordFreq"]
    fileDF = pd.read_csv(os.path.join(dataPath, fileName))
    stopWords = loadStopWords()

    rowList = []
    print(f"Extracting from {fileName}")
    for index, row in fileDF.iterrows():
        #Just some verbose output
        print(f"Extracting Index {index} out of {fileDF.shape[0]} "
            + f"({(index / fileDF.shape[0]) * 100:.3f}%)\r", end="\r")
        
        doc = spacy(str(row["Body"]))
        wordCounts = getWordCounts(doc)
        rowDict = defaultdict(lambda: None)

        numWords = totalNumWords(doc)
        
        #Adding features to a dict for CSV storage
        rowDict["userID"] = row["OwnerUserId"]
        rowDict["postID"] = row["Id"]
        rowDict["numWords"] = numWords
        rowDict["numChars"] = totalNumChars(doc)
        rowDict["yule"] = yuleify(wordCounts)
        rowDict["avgSentenceWords"] = avgSentenceWords(doc)
        rowDict["avgSentenceChars"] = avgSentenceChars(doc)

        #Don't you love division by 0?
        numWords = 1 if numWords == 0 else numWords

        #These are essentially meta-frequencies
        #So counting the number of words that appear once, twice, 3, 4, and 5 times
        rowDict["legoHapax"] = legomena(wordCounts) / numWords
        rowDict["legoDis"] = legomena(wordCounts, 2) / numWords
        rowDict["legoTris"] = legomena(wordCounts, 3) / numWords
        rowDict["legoTetrakis"] = legomena(wordCounts, 4) / numWords
        rowDict["legoPentakis"] = legomena(wordCounts, 5) / numWords

        #Putting all of the stop words as individual features
        stoppies = stopWordFreq(wordCounts)
        for stopWord, freq in stoppies.items():
            rowDict[f"stop-{stopWord}"] = freq / numWords

        rowList.append(rowDict)

    xTracted = pd.DataFrame(rowList)
    pprint(xTracted)
    print(f"{xTracted.shape[1]} different features")
    
    print("Completed Extraction")
    xTracted.to_csv(os.path.join(dataPath, fileName[:-4] + "Extracted.csv"))


if __name__ == "__main__":
    if debug: 
        main()
    else:
        if miniXtract:
            runExtraction("miniRestrictedPosts.csv")
        elif singleXtract:
            runExtraction("singlePost.csv")
        else:
            runExtraction("RestrictedPosts.csv")
