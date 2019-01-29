#bu user classification main file

from argparse import ArgumentParser

from characterisation.classification import sklearnHelper as skh
from characterisation.classification import sklearnSVM as svm

from characterisation.helpers.reduction import initAE

"""
Performs dimension reduction on the dataset

I've got an idea. Train the dimension reduction on a StackSite that we won't use.
Something like "Movies & TV" or "Personal Finance & Money" are good ones
For the time being we'll use serverfault for an MVP, since it's already downloaded.
"""
def dimReduction(dataset="worldbuilding", mini=False, folds=5):
    xTrain, yTrain, xTest, yTest = skh.split(dataset, mini, folds)

    convAE = initAE("serverfault", mini=mini)

    #Reducing the dimensions of the SVM training and testing data
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
Train a SVM on a reduced input. So input data which has been processed through
the Convolutional AutoEncoder.
"""
def reducedSVM(dataset="worldbuilding", mini=False, folds=5):
    dimReduction(dataset, mini, folds)

    return svm.initSVM(xTrain, yTrain)

"""
Given an input, predicts the appropriate equivalence class it belongs to.

Predicting equivalence classes on their own are pointless though.
"Oh look my writing style is the same as UserID #8291". Like, that tells you nothing.
Granted we shouldn't care if it tells humans nothing, as long as we can later work back
and transform text to look like it was made by 8291. For humans, we could show posts/comments
of a similar style, retrieved from users in the equivalence class.

To make it happy, xInput should be array-like
"""
def svmPredict(xInput):
    pass

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

    #From here we can predict equivalence classes by using the svm.predict() method
