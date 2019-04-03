import characterisation.classification.sklearnHelper as skh

from pprint import pprint

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV

from characterisation.classification.sklearnHelper import hyperSearch
from helpers import fileIO

"""
Initialises a sci-kit learn neural net based on the results of the hyper-parameter search.
"""
def initNet(trainX, trainY, paramDist=None, searchNum=20, fullSearch=False):
    netty = None

    if paramDist is None:
        paramDist = {
            "activation": ["relu", "identity", "logistic", "tanh"],
            "solver": ["lbfgs", "sgd", "adam"],
            "alpha": [0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
            "learning_rate": ["constant", "invscaling", "adaptive"],
            "learning_rate_init": [0.1, 0.01, 1e-3, 0.5e-3],
            "tol": [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
        }

    # preSearch = MLPClassifier(activation="relu")
    if fullSearch:
        model = skh.fullHyperSearch(MLPClassifier(probability=True), paramDist, trainX, trainY)
        fileIO.savePickle(model, "classyANN_FullSearch.pkl")
    else:
        model = hyperSearch(MLPClassifier(probability=True), paramDist, trainX, trainY, searchNum)
        fileIO.savePickle(model, f"classyANN_{searchNum}Searches.pkl")

    return model
    
def predictProbs(model, xInput):
    print("Predicted probabilities")
    pprint(model.predict_proba(xInput))
    return skh.predictProbs(model, xInput)

def classify():
    classy = MLPClassifier(activation="relu")
    # classy = MLPClassifier(activation="relu") implement using my extracted data 

if __name__ == "__main__":
    trainX, trainY, _, _ = skh.split()
    
    initNet(trainX, trainY)