#bu user classification main file

from argparse import ArgumentParser

from characterisation.classification import sklearnHelper as skh
from characterisation.classification import sklearnSVM as svm

from characterisation.reduction import initAE

"""
Performs dimension reduction on the dataset

I've got an idea. Train the dimension reduction on a StackSite that we won't use.
Something like "Movies & TV" or "Personal Finance & Money" are good ones
For the time being we'll use serverfault for an MVP, since it's already downloaded.
"""
def dimReduction(dataset="worldbuilding", mini=False, folds=5):
    xTrain, yTrain, xTest, yTest = skh.split(dataset, mini, folds)

    convAE = initAE("serverfault", mini=mini)

    xTrain = convAE.predict(xTrain)
    xTest = convAE.predict(xTest)

    #Need to figure out what these values are going to be
    #Are we just going to feed the previous xTrain, xTest arrays into the AE?
    #Or something else?
    #This is just for training btw
    #A real system will extract features and subsequently encode the input.
    #Okay, so a real system wouldn't return anything for yTrain, xTest, or yTest  
    return xTrain, yTrain, xTest, yTest

"""
This is the main file for classification of course, and this is, well, the main method.
Duh.

So, here, well at the command line, ideally a user will just enter in some text, it will
get classified by bu, and subsequently output users which resemble the input text. Done by 
our lovely equivalence classes of course. A stretch would be to display other posts/comments
of the same writing style.
"""
def main():
    myParser = ArgumentParser()

    myParser.add_argument("-d", "--dataset", dest="dataset", metavar="dataset", default="worldbuilding",
                        help="The dataset to use for classification purposes")

    myArgs = myParser.parse_args()

if __name__ == "__main__":
    xTrain, yTrain, xTest, yTest = dimReduction()

    mySVM() = svm.initSVM(xTrain, yTrain)
