from xml.dom import minidom
import xml.etree.ElementTree as etree

import io

from pprint import pprint

import pandas as pd

class ParseXML():
    def __init__(self, fileName):
        self.data = etree.parse(fileName)
        pprint(self.data)

    def parseRoot(self):
        """Return a list of dictionaries from the text
         and attributes of the children under this XML root."""
        return [self.parseElement(child) for child in iter(self.data)]

    """
    Modified from https://gist.github.com/mattmc3/712f280ec81044ec7bd12a6dda560787
    """
    def xml2df(self, idField="Id"):
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
    df = hi.xml2df()

    pprint(df.iloc[2])


if __name__ == "__main__":
    main()