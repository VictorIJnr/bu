import numpy as np

from pprint import pprint

from .sklearnHelper import pullData, filteredMap
from .sklearnHelper import hyperSearch, report
from .equiv import jumpy

from sklearn.svm import SVC

"""
This should retrieve the top x models found from the searching cross-validation
These models will the all be tested against the different equivalence class methods
(Jump Points, 95th Percentile of Users, Highest 90% of Scores)
"""
def pullTopX(num=5):
    pass

def initSVM():
    trainX, trainY, testX, testY = pullData()

    print(f"{len(np.unique(trainY))} different training classes\n\n")

    paramDist = {
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": list(range(6)),
        "gamma": ["auto", "scale", 0.01, 0.05, 0.1, 0.15, 0.2],
        "coef0": np.linspace(0, 1, num=21), #21 to accomodate for the endpoint (1)
        "shrinking": [True, False],
        "tol": [0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    }

    #I calculated it, doing a complete search with all of these parameters will take 
    #5 and a half days...
    classy = hyperSearch(SVC(probability=True), paramDist, trainX, trainY, searchNum=2)
    # classy = hyperSearch(SVC(), paramDist, trainX, trainY, searchNum=100)

    # print(type(classy.cv_results_))
    # pprint(classy.cv_results_)

    #This is a bad way of retaining the test data... I'll find another way to fix this sometime
    #Huh, I just thought of one, pass the training parameters to this method instead
    #So outside the method you still have the train/test split that you used.
    return classy, testX, testY

"""
Runs experiments on each of the different methods to determine equivalence classes
"""
def testEquiv():
    classy, testX, testY = initSVM()
    predictProbs = classy.predict_proba(testX)
    predictions = classy.predict(testX)

    mapping = filteredMap()
    print(f"User mapping {mapping}")
    print(f"Mapping length {len(mapping)}")
    print(f"{predictProbs.shape[1]} different classes")
    print(f"{len(np.unique(testY))} different actual classes")
    print("\n\n\n\n")

    
    print(f"Predictions {predictions}")
    print(f"First prediction: {predictions[0]}")
    print(f"Max value: {predictProbs[0]}")
    print(f"Prediction index: {np.argwhere(mapping==predictions[0])}")
    print(f"Max index: {np.argmax(predictProbs[0])}")
    print(f"Actual class: {testY[0]}")

    #Look at the predicted class then compare that to the maximum index in the probability array
    jumpy(predictProbs, testY)

if __name__ == "__main__":
    testEquiv()