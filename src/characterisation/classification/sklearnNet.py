from time import time

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV

def hyperSearch(paramDist=None, searchNum=20, trainX, trainY):
    if (paramDist is None):
        paramDist = {
            "activation": ["relu", "identity", "logistic", "tanh"],
            "solver": ["lbfgs", "sgd", "adam"],
            "alpha": [0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
            "learning_rate": ["constant", "invscaling", "adaptive"],
            "learning_rate_init": [0.1, 0.01, 1e-3, 0.5e-3],
            "tol": [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
        }
    
    preSearch = MLPClassifier(activation="relu")
    model = RandomizedSearchCV(preSearch, param_distribution=paramDist, n_iter=searchNum, cv=5)
    model.fit(trainX, trainY)

    return model
    

def classify():
    classy = MLPClassifier(activation="relu")
    # classy = MLPClassifier(activation="relu") implement using my extracted data 

def report(results, bestN=3):
    for i in range(1, bestN + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print(f"Model with rank: {i}")
            print(f"Mean validation score: {results["mean_test_score"][candidate]:.3f} "
                + f"(std: {results["std_test_score"][candidate]:.3f})")
            print(f"Parameters: {results["params"][candidate]}\n")

if __name__ == "__main__":
    pass