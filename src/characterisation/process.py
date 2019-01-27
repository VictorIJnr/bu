import os

from characterisation.stack import execRestrict
from characterisation.xTract import runExtraction

buPath = os.path.dirname(os.path.realpath(__file__))

def dataProcess(dataset, dataRows=None):
    execRestrict(dataset, dataRows)

    if dataRows is not None:
        runExtraction()