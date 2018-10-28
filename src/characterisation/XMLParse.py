from xml.dom import minidom
import xml.etree.ElementTree as etree

import io

from pprint import pprint

import pandas as pd

class ParseXML():
    def __init__(self, fileName):
        self.data = etree.parse(fileName)

    @staticmethod
    def convertXML(fileName="", idField="Id"):
        data = etree.parse(fileName)
        dataAsDict = []
        for i, child in enumerate(data.getroot()):
            entry = {}
            entry[child.attrib[idField]] = child.attrib
            dataAsDict.append(child.attrib)

        df = pd.DataFrame(dataAsDict)

        return df

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
    hi = ParseXML("../data/worldbuilding.stackexchange.com/Users.xml")
    df = hi.convert()
    df2 = ParseXML.convertXML("../data/worldbuilding.stackexchange.com/Posts.xml")

    pprint(df.iloc[2])


if __name__ == "__main__":
    main()