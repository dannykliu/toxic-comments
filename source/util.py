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
        extract toxic comment data from csv file
        returns a list of lists, where each sublist corresponds to one comment entry (and all its labels)
    """
    index = 0

    data = []   # list of lists, where each sublist is a line of the csv file

    with open(infile, 'rU') as fid:
        csvData = csv.reader(fid, delimiter=',')
    
        for line in csvData:
            if index!=0:
                data.append(line)
            index+=1

    return data

def extract(data, comment_col =1, predict_cols=[2,3,4,5,6,7]) :        
    """
    Parameters
    --------------------
        x      -- list of lists where each sublist is a line from a csv file

    Returns
    --------------------
        x      -- list of strings, length n (n comments)
        y      -- list of ints, 1 = toxic, 0 = non-toxic, labels for x
    """

    x = []
    y = []

    for line in data: 
        if any([int(line[predict_col]) == 1 for predict_col in predict_cols]):
            y.append(1)
        else: 
            y.append(0)
        x.append(line[comment_col])
    return x,y