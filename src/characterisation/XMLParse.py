import xml.etree.ElementTree as etree

import io
import os

from pprint import pprint

import pandas as pd

buPath = os.path.dirname(__file__)

class ParseXML():
    def __init__(self, filePath):
        filePath = os.path.join(buPath, filePath)
        self.data = etree.parse(filePath)

    @staticmethod
    def convertAll(dirPath=""):
        dfs = []
        for fileName in os.listdir(os.path.join(buPath, dirPath)):
            dfs.append(ParseXML.convertXML(os.path.join(dirPath, fileName)))

        return dfs

    @staticmethod
    def convertAllToCSV(dirPath=""):
        dfs = []
        for fileName in os.listdir(os.path.join(buPath, dirPath)):
            df = ParseXML.convertXML(os.path.join(dirPath, fileName))
            dfs.append(df)
            ParseXML.dfToCSV(df, fileName, dirPath)

        return dfs

    @staticmethod
    def convertXML(filePath="", idField="Id"):
        pprint("Converting " + filePath + " to a dataFrame.")

        filePath = os.path.join(buPath, filePath)
        data = etree.parse(filePath)
        dataAsDict = []
        for i, child in enumerate(data.getroot()):
            entry = {}
            entry[child.attrib[idField]] = child.attrib
            dataAsDict.append(child.attrib)

        df = pd.DataFrame(dataAsDict)

        return df
    
    @staticmethod
    def dfToCSV(df, fileName, filePath):
        pprint("Converting " + fileName + " into a csv")
        df.to_csv(os.path.join(filePath, fileName)[3:-3] + "csv")

    """
    Modified from https://gist.github.com/mattmc3/712f280ec81044ec7bd12a6dda560787
    """
    def convert(self, idField="Id"):
        dataAsDict = []
        for i, child in enumerate(self.data.getroot()):
            entry = {}
            entry[child.attrib[idField]] = child.attrib
            dataAsDict.append(child.attrib)

        df = pd.DataFrame(dataAsDict)

        return df
    

    def processData(self):
        """ Initiate the root XML, parse it, and return a dataframe"""
        return pd.DataFrame(list(self.iter_docs(self.data.getroot())))

def main():
    dfs = ParseXML.convertAllToCSV("../../data/worldbuilding.stackexchange.com/")


if __name__ == "__main__":
    main()