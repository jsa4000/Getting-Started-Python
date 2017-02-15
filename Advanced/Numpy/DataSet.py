import numpy as np
import os.path
import xml.etree.ElementTree as ET

__all__ = [
    'getDataSet' 
    ]

def iter_DataSet(root):
    for dataset in root.iterfind('.//DataSet'):
        resultset = []
        for datapoint in dataset.iterfind('.//DataPoint'):
            doc_Inputs = []
            inputs = datapoint.iterfind('.//Input').next()
            for input in inputs.iterfind('.//Data'):
                doc_Inputs.append ( input.text)
            doc_Outputs = []
            outputs = datapoint.iterfind('.//Output').next()
            for output in outputs.iterfind('.//Data'):
                doc_Outputs.append ( output.text)
            resultset.append([doc_Inputs, doc_Outputs])            
        yield resultset


def getDataSet(filePath, dataset = -1):
    """
        XML Sample: ".\NeuralNetwork\data_sets.xml"
            <Root>
              <DataSet>
                <DataPoint>
                  <Input>
                    <Data Index="0">0</Data>
                    <Data Index="1">0</Data>
		             ...
		            <Data Index="23">0</Data>
		            <Data Index="24">0</Data>
                  </Input>
                  <Output>
                    <Data Index="0">1</Data>
		            <Data Index="1">0</Data>
		            <Data Index="2">0</Data>
		            <Data Index="3">0</Data>
                  </Output>
                </DataPoint>
                ...
              </DataSet>
            </Root>

        REturn Inputs , Outputs for the current dataset
    
    """
    # Check if the file exist
    if (os.path.isfile(filePath)):
        try:
            #Parse the document
            tree = ET.parse(filePath) # or ET.fromstring(xml_string) from a string
            datasets = list(iter_DataSet(tree.getroot()))
       
            if (dataset == -1 and len(datasets) > 0):
                # Returns the first element in the data sets    
                return list(datapoint[0] for datapoint in datasets[0]), list(datapoint[1] for datapoint in datasets[0])  
            elif (len(datasets) > dataset):
                return list(datapoint[0] for datapoint in datasets[dataset]), list(datapoint[1] for datapoint in datasets[dataset])
            else:
                return None, None
        except Exception as ex:
            # Error while parsing the document
            return None, None
    # If the file doesn exist or error
    return None, None
