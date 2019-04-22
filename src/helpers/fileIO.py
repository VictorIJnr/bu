import os
import json, pickle

buPath = os.path.dirname(os.path.realpath(__file__))

def savePickle(data, fileName):
    filePath = os.path.join(buPath, "myPickles")
    if not os.path.exists(filePath):
        os.makedirs(filePath)

    with open(os.path.join(filePath, fileName), "wb") as myPickles:
        pickle.dump(data, myPickles)

def loadPickle(fileName):
    try:
        with open(os.path.join(buPath, "myPickles", fileName), "rb") as myPickles:
            return pickle.load(myPickles)
    except:
        raise FileNotFoundError("Pickle file does not exist")

def saveJSON(data, fileName):
    filePath = os.path.join(buPath, "myJSONs")
    if not os.path.exists(filePath):
        os.makedirs(filePath)

    with open(os.path.join(filePath, fileName), "w") as myJSON:
        json.dump(data, myJSON)

def loadJSON(fileName):
    with open(os.path.join(buPath, "myJSONs", fileName), "r") as myJSON:
        return json.load(myJSON)