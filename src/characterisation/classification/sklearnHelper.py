import numpy as np

from time import time

from sklearn.model_selection import RandomizedSearchCV

def hyperSearch(searchModel, paramDist, trainX, trainY, searchNum=20, verbose=True, cv=5):
    model = RandomizedSearchCV(searchModel, param_distributions=paramDist, n_iter=searchNum, cv=cv)
    
    start = time()
    model.fit(trainX, trainY)

    print("") #Only keep this whilst I have all those stupid warnings

    if verbose:
        print(f"Parameter search took {time() - start:.2f} seconds to explore {searchNum}"
            + " possibilities")
        report(model.cv_results_)

    return model


"""
Runs an exhaustive search through all the paramaters specified in the parameter distribution
"""
def fullHyperSearch(searchModel, paramDist, trainX, trainY, verbose=True, cv=5):
    model = GridSearchCV(searchModel, param_distributions=paramDist, cv=cv)

    start = time()
    model.fit(trainX, trainY)
    
    if verbose:
        print(f"Parameter search took {time() - start:.2f} seconds to explore {searchNum}"
            + " possibilities")
        report(model.cv_results_)
    
    return model

def report(results, bestN=3):
    for i in range(1, bestN + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print(f"Model with rank: {i}")
            print(f"Mean validation score: {results['mean_test_score'][candidate]:.3f} "
                + f"(std: {results['std_test_score'][candidate]:.3f})")
            print(f"Parameters: {results['params'][candidate]}\n")