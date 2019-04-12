import os
import re
import sys

import spacy as sp
import pandas as pd

from collections import defaultdict
from pprint import pprint

from helpers import fileIO

worldbuilding = "worldbuilding.stackexchange.com"
serverfault = "serverfault.com"

spacy = sp.load("en")
limitRows = True
# rowLimit = 64
# rowLimit = 288
rowLimit = 512

dataset = worldbuilding
buPath = os.path.dirname(os.path.realpath(__file__))
dataPath = os.path.join(buPath, "..", "..", "..", "data")

def cleanBody(df, index, row):
    oldBody = row["Body"]
    cleaner = re.compile("<.*?>|\r?\n|\r")
    df.at[index, "Body"] = re.sub(cleaner, "", str(oldBody))

def restrict(df, commentDF, limit=5, myDataset=dataset):
    userDict = preRestriction(df, commentDF, limit, myDataset)
    # userDict = miniPreRestriction(df)

    initNumUsers = len(userDict)
    print(f"Number of pre-filtered users:\t{initNumUsers}")
    
    userDict = {user: count for (user, count) in userDict.items() if count >= limit}
    print("\nGenerated the filtered user dictionary.")
    print(f"Number of filtered users:\t{len(userDict)}\n")

    newDF = df.query("OwnerUserId in @userDict.keys()")

    return newDF, userDict, initNumUsers

def preRestriction(df, commentDF, limit=5, myDataset=dataset):
    userDict = defaultdict(lambda: 0)

    try:
        #Load a pre-filtered dataset if it exists
        print("Attempting to use cached prefiltered dataset...")
        userDict = fileIO.loadJSON(f"preRestrict_{myDataset.split('.')[0]}.json")
    except FileNotFoundError:
        #If it didn't exist, build it
        print("Pre-existing prefiltered dataset does not exist. Building from scratch...")
        
        for index, row in df.iterrows():
            sys.stdout.write(f"Progress:\t{(index / df.shape[0]) * 100:.3f}%\r")
            sys.stdout.flush()
            
            #If the post by the user is an answer
            if (row["PostTypeId"] == 2):
                cleanBody(df, index, row)
                userDict[row["OwnerUserId"]] += 1
            #If the post is a question extra conditions must be met 
            elif (row["PostTypeId"] == 1):
                count = 0
                #Getting all of the comments for the current post
                postID = row["Id"]
                postComments = commentDF.query("PostId == @postID")

                #Counting the number of comments made by the question creator
                for commentIndex, comment in postComments.iterrows():
                    if (comment["UserId"] == row["OwnerUserId"]):
                        count += 1
                #Only if the question creator has a sufficient amount of comments under
                #their question, are they considered to have enough data
                if count > limit:
                    cleanBody(df, index, row)
                    userDict[row["OwnerUserId"]] += 1
        #Saving the prefiltered dataset                    
        fileIO.saveJSON(userDict, f"preRestrict_{myDataset.split('.')[0]}.json")
    else:
        print("Cached prefiltered dataset located...")
    
    return userDict

def miniPreRestriction(df):
    userDict = defaultdict(lambda: 0)

    for index, row in df.iterrows():
        sys.stdout.write(f"Progress:\t{(index / df.shape[0]) * 100:.3f}%\r")
        sys.stdout.flush()

        oldBody = row["Body"]
        cleaner = re.compile("<.*?>|\r?\n|\r")
        df.at[index, "Body"] = re.sub(cleaner, "", str(oldBody))

        userDict[row["OwnerUserId"]] += 1 
    return userDict

def main():
    print("Loading data into dataframe...")
    postDF = pd.read_csv(os.path.join(dataPath, dataset, "Posts.csv"))
    commentDF = pd.read_csv(os.path.join(dataPath, dataset, "Comments.csv")) 

    initNumPosts = postDF.shape[0]
    print(f"Initial number of Posts:\t{initNumPosts}")

    postDF, allUsers, initNumUsers = restrict(postDF, commentDF)
    print(f"Number of Users:\t{len(allUsers)}")
    print(f"Filtered Posts size:\t{postDF.shape[0]}")

    print(f"\n{initNumUsers - len(allUsers)} removed users.")
    print(f"{initNumPosts - postDF.shape[0]} removed posts.")

    postDF = postDF[:rowLimit] if limitRows else postDF
    savePath = "miniRestrictedPosts.csv" if limitRows else "RestrictedPosts.csv"
    postDF.to_csv(os.path.join(dataPath, dataset, savePath))

"""
Filters a dataset to only include users which have reached a provided threshold
"""
def execRestrict(myDataset, limit=rowLimit):
    allPosts = None
    allComments = None

    execDataset = None

    if myDataset == "worldbuilding":
        execDataset = worldbuilding
        dataset = worldbuilding
    elif myDataset == "serverfault":
        execDataset = serverfault
        dataset = serverfault

    print("Loading data into dataframe...")
    #Load a cached dataframe pickle file if it exists
    try:
        print("Attempting to load a cached dataframe...")
        allPosts = fileIO.loadPickle(f"dataframe_{dataset.split('.')[0]}_posts.pkl")
        allComments = fileIO.loadPickle(f"dataframe_{dataset.split('.')[0]}_comments.pkl")
    except FileNotFoundError:
        print("A cached dataframe could not be found.\nBuilding from scratch...\n")
        allPosts = pd.read_csv(os.path.join(dataPath, dataset, "Posts.csv"))
        allComments = pd.read_csv(os.path.join(dataPath, dataset, "Comments.csv"))

        fileIO.savePickle(allPosts, f"dataframe_{dataset.split('.')[0]}_posts.pkl")
        fileIO.savePickle(allComments, f"dataframe_{dataset.split('.')[0]}_comments.pkl")
    else:
        print("Cached dataframe located...\n")

    print(dataset.split('.')[0])
    print(f"Initial number of Posts:\t{allPosts.shape[0]}")

    #! Uncomment this when I'm done getting output for the report
    #// newPosts, _, _ = restrict(allPosts, allComments, myDataset=execDataset)

    #! The following line is just for my report
    newPosts, userDict, initNumUsers = restrict(allPosts, allComments, myDataset=execDataset)    

    newPosts = newPosts[:limit] if limit is not None else newPosts
    savePath = "miniRestrictedPosts.csv" if limit is not None else "RestrictedPosts.csv"
    newPosts.to_csv(os.path.join(dataPath, dataset, savePath))

    #! The following output was just for me to write my report
    print(f"Initial number of users:\t{initNumUsers}")
    print(f"Final number of Posts:\t{newPosts.shape[0]}")
    print(f"Final number of users:\t{len(userDict)}")

if __name__ == "__main__":
    main()