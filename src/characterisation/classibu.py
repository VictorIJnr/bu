#bu user classification main file

import numpy as np

from argparse import ArgumentParser
from pprint import pprint

from characterisation.classification import sklearnHelper as skh
from characterisation.classification import sklearnSVM as svm

from characterisation.classification.equiv import Equivs
from characterisation.helpers.reduction import initAE, initDatasetAE

"""
Performs dimension reduction on the dataset

I've got an idea. Train the dimension reduction on a StackSite that we won't use.
Something like "Movies & TV" or "Personal Finance & Money" are good ones
For the time being we'll use serverfault for an MVP, since it's already downloaded.
"""
def dimReduction(dataset="worldbuilding", mini=False, folds=5):
    xTrain, yTrain, xTest, yTest = skh.split(dataset, mini, folds)

    #"serverfault" could be an argument, but it's fine being constant
    convAE = initDatasetAE(dataset="serverfault", mini=mini, factor=1)

    xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))
    xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))

    #Reducing the dimensions of the SVM training and testing data
    xTrain = convAE.predict(xTrain)
    xTest = convAE.predict(xTest)

    #This reshaping is to ensure that the SVM can be trained
    #The third dimension is the number of filters, so I should be fine dropping this.
    xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1]))
    xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1]))
    
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
def reducedSVM(dataset="worldbuilding", mini=True, folds=5, load=False, returnTest=False, 
                searchNum=5, full=False):
    xTrain, yTrain, xTest, yTest = dimReduction(dataset, mini, folds)

    if returnTest:
        return svm.initSVM(xTrain, yTrain, loadModel=load, searchNum=searchNum, fullSearch=full), xTest, yTest
    else:
        return svm.initSVM(xTrain, yTrain, loadModel=load, searchNum=searchNum, fullSearch=full)

"""
Train an SVM on regular input. Retains the complete 304 dimensions by skipping dimension
reduction with the convolutional autoencoder.
"""
def skippedSVM(dataset="worldbuilding", mini=True, folds=5, load=False, returnTest=False, 
                searchNum=5, full=False):
    xTrain, yTrain, xTest, yTest = skh.split(dataset, mini, folds)
    
    if returnTest:
        return svm.initSVM(xTrain, yTrain, loadModel=load, searchNum=searchNum, fullSearch=full), xTest, yTest
    else:
        return svm.initSVM(xTrain, yTrain, loadModel=load, searchNum=searchNum, fullSearch=full)

"""
Given an input, predicts the appropriate equivalence class it belongs to.

Predicting equivalence classes on their own are pointless though.
"Oh look my writing style is the same as UserID #8291". Like, that tells you nothing.
Granted we shouldn't care if it tells humans nothing, as long as we can later work back
and transform text to look like it was made by 8291. For humans, we could show posts/comments
of a similar style, retrieved from users in the equivalence class.

To make it happy, xInput should be array-like
Passing in a model is not an ideal solution. I should have my own SVM class which has a predict
method to specify the equivalence class method and input to predict.
"""
def svmPredict(model, xInput, equivClass=Equivs.JUMP):
    return svm.predict(model, xInput, equivClass)

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

    myParser.add_argument("--dataset", "-d", default="worldbuilding",
                        help="The dataset to use for classification purposes")
    myParser.add_argument("--train", "-t", default=False, action="store_true",
                        help="Debug argument to train the SVM on a reduced dataset. Default: True")
    myParser.add_argument("--mini", "-m", default=False, action="store_true",
                        help="Whether a small subsection of the dataset will be used for training."
                            + "Default: False")
    myParser.add_argument("--folds", default=5, type=int,
                        help="The number of folds used for cross-fold validation.")

    myArgs = myParser.parse_args()

    if myArgs.train:
        reducedSVM(myArgs.dataset, myArgs.mini, myArgs.folds)

if __name__ == "__main__":
    main()