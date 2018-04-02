"""
Authors: Sasha Friedrich
         Danny Liu
         Amelia Sheppard
         Dalton Varney
Date:    4/1/18
Description: Useful functions
"""

import numpy as np

import csv

def load(infile):
    """
    """
    index = 0

    data = []   # list of lists, where each sublist is a line of the csv file

    with open(infile, 'rU') as fid:
        csvData = csv.reader(fid, delimiter=',')
        data =[]
    
        for line in csvData:
            data.append(line)

    return np.array(data)
