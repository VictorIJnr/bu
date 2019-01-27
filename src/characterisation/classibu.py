#bu user classification

from characterisation.classification import sklearnHelper as skh
from characterisation.classification import sklearnSVM as svm

from characterisation.reduction import initAE

"""
Performs dimension reduction on the dataset
"""
def dimReduction():
    xTrain, yTrain, xTest, yTest = skh.pullData(fullData=True)

    convAE = initAE(xTrain, xTest)

    xTrain = convAE.predict(xTrain)
    xTest = convAE.predict(xTest)

    #Need to figure out what these values are going to be
    #Are we just going to feed the previous xTrain, xTest arrays into the AE?
    #Or something else?
    #This is just for training btw
    #A real system will extract features and subsequently encode the input 
    return xTrain, yTrain, xTest, yTest


if __name__ == "__main__":
    xTrain, yTrain, xTest, yTest = dimReduction()

    mySVM() = svm.initSVM(xTrain, yTrain)
