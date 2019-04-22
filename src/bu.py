#Main bu file

import numpy as np

from argparse import ArgumentParser

from characterisation.helpers.xTract import extractText
from characterisation.classification.equiv import Equivs

import characterisation.classibu as classibu

def main(myArgs):
    classiSVM = classibu.reducedSVM(load=True)
    
    extractedText = np.array(extractText(myArgs.input))

    prediction = classibu.svmPredict(classiSVM, extractedText, Equivs.SCORE_DIST)

    outputString = "There are no other observed users of a comparable writing style to you."\
        + "You're truly one of a kind!" if len(prediction) == 0 else\
        f"These users have a similar writing style to you: {prediction}"
    
    print(outputString)

if __name__ == "__main__":
    myParser = ArgumentParser()

    myParser.add_argument("--input", default="Welcome to User Stylometry Association!",
                            help="Provide input to be classified.")

    myArgs = myParser.parse_args()
    
    main(myArgs)