"""
Authors: Sasha Friedrich
         Amelia Sheppard
Date: 4/1/18
Description: 
"""

import numpy as np

import csv

from util import *   # data extraction functions

import math

import matplotlib as mpl
import matplotlib.pyplot as plt

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

def plot_histogram(X, y, Xname, yname, bins=None) :
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
    if bins==None: 
        features = set(X)
        nfeatures = len(features)
        test_range = range(int(math.floor(min(features))), int(math.ceil(max(features)))+1)
        if nfeatures < 10 and sorted(features) == test_range:
            bins = test_range + [test_range[-1] + 1] # add last bin
            align = 'left'
        else :
            bins = 10
            align = 'mid'
    else: 
        align = 'left'
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

def countWords(X, testWords): 
    """ counts the number of occurrences of particular words

    Parameters
    -----------
        X - list of comments (strings), length n

    Returns
    -------
        y - list of counts of occurences of words, length n
    """

    y = []
    for string in X: 
        testWordCount = 0
        words = extract_words(string)
        for testWord in testWords: 
            for word in words: 
                if testWord == word: 
                    testWordCount+=1
        y.append(testWordCount)

    return y

def split_comments(X):
    """
    Parameters
    --------------------
        X       -- list of comments

    Return 
    --------------------
        new_X   -- list of comments where each comment is a list of words
    """
    new_X = []
    for comment in X:
        new_X.append(extract_words_nolower(comment))
    return new_X

def visualize_capitalization(X, y):
    """
    Parameters
    --------------------
        X       -- list of comments (list of comments)
        y       -- list of associated labels (toxic vs nontoxic)

    Make a histogram of capitalization percentages for toxic vs nontoxic comments.

    """
    #capitalizaiton percentage
    cap_per = []
    for comment in X:
        stripped_comment = comment.replace(" ", "")
        cap_count = 0
        uppers = [l for l in stripped_comment if l.isupper()]
        percentage = 1.0*len(uppers)/len(stripped_comment)
        cap_per.append(percentage)
    
    plot_histogram(cap_per, y, "Capitalization Percentage", "Toxicity")

def visualize_comment_length(X, y):
    """
    Parameters
    --------------------
        X       -- list of comments (list of comments)
        y       -- list of associated labels (toxic vs nontoxic)

    Make a histogram of comment length for toxic vs nontoxic comments.

    """
    #comment length
    comment_length = []
    for comment in X:
        comment_length.append(len(comment))

    plot_histogram(comment_length, y, "Comment Length", "Toxicity")
    
def main(): 
    raw_data = load('../data/features_subsample.csv')
    x,y=extract(raw_data) # x is list of comments, y is associated labels

    swear_words = ['shit', 'fuck', 'damn', 'bitch', 'crap', 'piss', 'ass', 'asshole', 'bastard']
    # swear_counts = np.asarray(countWords(x, swear_words))
    # assert len(swear_counts) == len(y)
    # plot_histogram(swear_counts, np.asarray(y), 'number of swear words', 'toxicity', bins = [0, 1, 2, 3, 4, 5,6,7])

    sex_words = ['dick', 'suck', 'pussy', 'cunt', 'penis', 'balls', 'testicles', 'pubic', 'genitals', 'sex', 'fuck','sex']
    counts = np.asarray(countWords(x, sex_words+swear_words))
    plot_histogram(counts, np.asarray(y), 'number of swear/sex-related words', 'toxicity', bins = [0, 1, 2, 3, 4, 5,6,7])


if __name__ == "__main__":
    main()
