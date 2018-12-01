from time import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

def hyperSearch(searchModel=None, paramDist=None, searchNum=20, trainX, trainY):
    if (paramDist is None):
        return
    
    model = RandomizedSearchCV(searchModel, param_distribution=paramDist, n_iter=searchNum, cv=5)
    model.fit(trainX, trainY)

    return model
    

def classify():
    paramDist = {
        "n_estimators": [5..20]
    }
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