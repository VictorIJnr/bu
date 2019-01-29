import os

from argparse import ArgumentParser

from characterisation.stack import execRestrict
from characterisation.xTract import execXtract

buPath = os.path.dirname(os.path.realpath(__file__))
dataPath = os.path.join(buPath, "..", "..", "data")

worldbuilding = "worldbuilding.stackexchange.com"
serverfault = "serverfault.com"

"""
Pull a dataset to extract features
"""
def dataProcess(dataset, fileType, dataRows=None):
    execRestrict(dataset, limit=dataRows)

    print(f"Nony data rows {dataRows}")
    
    execXtract(pullDatasetFile(dataset, fileType, mini=(dataRows is not None)))

"""
Returns the file name of a given dataset file as a string
"""
def pullDatasetFile(dataset, fileType, mini=False):
    myDataset = None

    if dataset == "worldbuilding":
        myDataset = worldbuilding
    elif dataset == "serverfault":
        myDataset = serverfault

    return os.path.join(myDataset, pullFileType(myDataset, fileType, mini))

"""
Pulls the appropriate file for feature extraction
"""
def pullFileType(dataset, fileType, mini=False):
    #I really need to use enums here but I have no idea how to do them in Python
    #There has to be a better way to do this right? This seems dumb
    retValue = None
    if fileType == "Posts":
        retValue = "RestrictedPosts.csv" 
    elif fileType == "Comments":
        retValue = "RestrictedComments.csv"

    return retValue if not mini else "mini" + retValue

if __name__ == "__main__":
    argParser = ArgumentParser()

    #This can be expanded for multiple different datasets
    argParser.add_argument("-d", "--dataset", dest="dataset", metavar="dataset", default="worldbuilding",
                        help="The dataset to run feature extraction")
    #This can also be expanded for multiple filetypes
    argParser.add_argument("-f", "--filetype", dest="fileType", metavar="fileType", default="Posts",
                        help="The file to extract features")

    argParser.add_argument("-l", "--limit", dest="limit", metavar="limit", type=int,
                        help="Sets a limit on the number of rows to extract from the dataset")


    myArgs = argParser.parse_args()

    #Use command line arguments here 
    dataProcess(myArgs.dataset, myArgs.fileType, dataRows=myArgs.limit)