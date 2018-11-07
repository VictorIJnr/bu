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

dataset = serverfault
buPath = os.path.dirname(os.path.realpath(__file__))
dataPath = os.path.join(buPath, "..", "..", "data")

postDF = pd.read_csv(os.path.join(dataPath, dataset, "Posts.csv"))
commentDF = pd.read_csv(os.path.join(dataPath, dataset, "Comments.csv"))

def restrict(df, limit=5):
    global postDF
    userDict = defaultdict(lambda: 0)
    """
    for index, row in df.iterrows():
        sys.stdout.write(f"Progress:\t{(index / postDF.shape[0]) * 100:.3f}%\r")
        sys.stdout.flush()
        
        #If the post by the user is an answer
        if (row["PostTypeId"] == 2):
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
                userDict[row["OwnerUserId"]] += 1
    """
    for index, row in df.iterrows():
        sys.stdout.write(f"Progress:\t{(index / df.shape[0]) * 100:.3f}%\r")
        sys.stdout.flush()

        userDict[row["OwnerUserId"]] += 1

    print(f"Number of pre-filtered users:\t{len(userDict)}")
    
    userDict = {user: count for (user, count) in userDict.items() if count >= limit}
    print("\nGenerated the filtered user dictionary.")
    print(f"Number of filtered users:\t{len(userDict)}\n")

    newDF = df.query("OwnerUserId in @userDict.keys()")

    cleaner = re.compile("<.*?>")
    newDF["Body"].apply(lambda body: re.sub(cleaner, '', str(body)))

    return newDF, userDict

def main():
    global postDF
    print(f"Initial Size:\t{postDF.shape[0]}")

    postDF, allUsers = restrict(postDF)
    print(f"Number of Users:\t{len(allUsers)}")
    print(f"Filtered Posts size:\t{postDF.shape[0]}")

    postDF.to_csv(os.path.join(dataPath, dataset, "RestrictedPosts.csv"))

if __name__ == "__main__":
    main()