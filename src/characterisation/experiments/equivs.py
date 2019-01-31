#Running experiments for different equivalence classes
#May also generate appropriate graphs as a result

from argparse import ArgumentParser

from characterisation.classibu import reducedSVM

def initSVM(xTrain, yTrain):
    reducedSVM(xTrain, yTrain)

"""
Okay. So this will run experiments for different equivalence classes.
So: Jump Points, Score Distributions, and User Percentiles.

The model type used (SVM, RFs, ANNs, etc.) shouldn't matter too much. For now I'll
just be using SVMs since that's the only one properly implemented.

The experiments should be run multiple times on unseen data - so I'll probs need a new
dataset - on a model which has been sufficiently trained on extracted features which have
been reduced. Through the ConvAE ofc.  
"""
def main():
    mySVM = reducedSVM(mini=False, load=True)
    
    pass

if __name__ == "__main__":
    main()