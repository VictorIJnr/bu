import numpy as np

def jumpy(classPreds, targetIndeces):
    firstClass = targetIndeces[0]
    firstProbs = classPreds[0]

    #Loop through all of the predictions and their corresponding actual values
    for i in range(classPreds.shape[0]):
        pass

    #The probability calculated for the actual chosen class
    print("\n\n\n\n\n\n\n")
    print(firstClass)

    #ALSO the value of first class doesn't refer to the index, it's value could be outside
    #The range
    #I could fix this by somehow creating a tuple between classes and their probabilities
    #But scikit will need to have a method which could help me
    firstClassProb = firstProbs[int(firstClass)]

    firstProbs = np.sort(firstProbs)[::-1]

    diffs = np.diff(firstProbs)

    #This isn't future proof, even though it's unlikely, multiple differences may have the
    #same value, so this could return either an array or a scalar
    firstMax = np.argmax(diffs)

    print(firstProbs)
    print(diffs)

    print(diffs.shape)
    print(np.argmax(diffs))

    print("First Max")
    print(type(firstMax))
    firstProbs = firstProbs[:int(firstMax)]

    #Check if the actual class is contained within these probabilities
    #Which would need like a tuple or a dict key value mapping
