import os
import re
import sys

import spacy as sp
import pandas as pd

from collections import defaultdict
from pprint import pprint

from characterisation.XMLParse import ParseXML

worldbuilding = "worldbuilding.stackexchange.com"
serverfault = "serverfault.com"

spacy = sp.load("en")
limitRows = False
rowLimit = 256

dataset = worldbuilding
buPath = os.path.dirname(os.path.realpath(__file__))
dataPath = os.path.join(buPath, "..", "..", "data")

print("Loading data into dataframe...")
postDF = pd.read_csv(os.path.join(dataPath, dataset, "Posts.csv"))
commentDF = pd.read_csv(os.path.join(dataPath, dataset, "Comments.csv"))

def cleanBody(df, index, row):
    oldBody = row["Body"]
    cleaner = re.compile("<.*?>|\r?\n|\r")
    df.at[index, "Body"] = re.sub(cleaner, "", str(oldBody))

def restrict(df, limit=5):
    global postDF
    userDict = preRestriction(df, limit)
    # userDict = miniPreRestriction(df)

    initNumUsers = len(userDict)
    print(f"Number of pre-filtered users:\t{initNumUsers}")
    
    userDict = {user: count for (user, count) in userDict.items() if count >= limit}
    print("\nGenerated the filtered user dictionary.")
    print(f"Number of filtered users:\t{len(userDict)}\n")

    newDF = df.query("OwnerUserId in @userDict.keys()")

    return newDF, userDict, initNumUsers

def preRestriction(df, limit=5):
    userDict = defaultdict(lambda: 0)

    for index, row in df.iterrows():
        sys.stdout.write(f"Progress:\t{(index / postDF.shape[0]) * 100:.3f}%\r")
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
    global postDF

    initNumPosts = postDF.shape[0]
    print(f"Initial number of Posts:\t{initNumPosts}")

    postDF, allUsers, initNumUsers = restrict(postDF)
    print(f"Number of Users:\t{len(allUsers)}")
    print(f"Filtered Posts size:\t{postDF.shape[0]}")

    print(f"\n{initNumUsers - len(allUsers)} removed users.")
    print(f"{initNumPosts - postDF.shape[0]} removed posts.")

    postDF = postDF[:rowLimit] if limitRows else postDF
    savePath = "miniPosts.csv" if limitRows else "RestrictedPosts.csv"
    postDF.to_csv(os.path.join(dataPath, dataset, savePath))

if __name__ == "__main__":
    main()