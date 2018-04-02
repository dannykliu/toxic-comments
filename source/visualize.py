"""
Authors: Sasha Friedrich
         Amelia Sheppard
Date: 4/1/18
Description: 
"""

import numpy as np

import csv

from util import *

def lineplot(x, y, label):
    """
    Make a line plot.
    
    Parameters
    --------------------
        x            -- list of doubles, x values
        y            -- list of doubles, y values
        label        -- string, label for legend
    """
    
    xx = range(len(x))
    plt.plot(xx, y, linestyle='-', linewidth=2, label=label)
    plt.xticks(xx, x)    
    plt.show()

def plot_histogram(X, y, Xname, yname) :
    """
    Author: Prof. Wu
    Plots histogram of values in X grouped by y.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """
    
    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in xrange(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))
    
    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = range(int(math.floor(min(features))), int(math.ceil(max(features)))+1)
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'
    
    # plot
    plt.figure()
    n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
    plt.xlabel(Xname)
    plt.ylabel('Frequency')
    plt.legend() #plt.legend(loc='upper left')
    plt.show()


def plot_scatter(X, y, Xnames, yname):
    """
    Author: Prof. Wu
    Plots scatter plot of values in X grouped by y.
    
    Parameters
    --------------------
        X      -- numpy array of shape (n,2), feature values
        y      -- numpy array of shape (n,), target classes
        Xnames -- tuple of strings, names of features
        yname  -- string, name of target
    """

    # plot
    targets = sorted(set(y))
    plt.figure()

    for target in targets :
        ### ========== TODO : START ========== ###
        
        xData=[]
        yData=[]
        for index in range(len(y)):
            if y[index] == target:
                xData.append(X[0][index]) 
                yData.append(X[1][index])
        plt.plot(xData,yData, '.', label = "survival = " + str(target))
        ### ========== TODO : END ========== ###
    plt.autoscale(enable=True)
    plt.xlabel(Xnames[0])
    plt.ylabel(Xnames[1])
    plt.legend()
    plt.show()

######################################################################
# data extraction functions
######################################################################

# def load(infile):
#     """
#     """
#     index = 0

#     data = []   # list of lists, where each sublist is a line of the csv file

#     with open(infile, 'rU') as fid:
#         csvData = csv.reader(fid, delimiter=',')
#         data =[]

#         for line in csvData:
#             data.append(line)

#     return np.array(data)

# def extract(data, comment_col =1, predict_cols=[2,3,4,5,6,7]) :        
#     """

#     Returns
#     --------------------
#         x      -- numpy array of shape (n,), feature values
#         y      -- numpy array of shape (n,), target classes
#         Xnames -- tuple of strings, names of features
#         yname  -- string, name of target
#     """

#    if any([int(line[predict_col]) == 1 for predict_col in predict_cols]):
#                     toxic_indices.append(index)
#                 else: 
#                     nontoxic_indices.append(index)
#         for line in csvData:
#             data.append(line)


def main(): 
    data = load('../data/subsampled_train.csv')
    print data.shape

if __name__ == "__main__":
    main()
