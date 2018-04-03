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
from string import punctuation
from collections import defaultdict
import json


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


def extract(data, comment_col=1) :
    """
    Parameters
    --------------------
        x      -- list of lists where each sublist is a line from a csv file

    Returns
    --------------------
        x      -- list of strings, length n (n comments)
        y      -- list of ints, 1 = toxic, 0 = non-toxic, labels for x
    """
    predict_cols = [2, 3, 4, 5, 6, 7]
    x = []
    y = []

    for line in data: 
        if any([int(line[predict_col]) == 1 for predict_col in predict_cols]):
            y.append(1)
        else: 
            y.append(0)
        x.append(line[comment_col])
    return x, y


def extract_words(input_string):
    """
    Processes the input_string, separating it into "words" based on the presence
    of spaces, and separating punctuation marks into their own words.

    Parameters
    --------------------
        input_string -- string of characters

    Returns
    --------------------
        words        -- list of lowercase "words"
    """

    for c in punctuation:
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()

def extract_words_nolower(input_string):
    """
    Processes the input_string, separating it into "words" based on the presence
    of spaces, and separating punctuation marks into their own words.

    Parameters
    --------------------
        input_string -- string of characters

    Returns
    --------------------
        words        -- list of lowercase "words"
    """

    for c in punctuation:
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.split()

def create_and_write_dictionary(datafile, bagfile):
    """
    Create dictionary from all words in training data and write to text file
    """
    raw_data = load(datafile)
    comments, y = extract(raw_data)
    word_list = defaultdict(int)
    for comment in comments:
        words = extract_words(comment)
        for word in words:
            word_list[word] += 1
    print len(word_list)
    new_word_list = defaultdict(int)
    for key in word_list:
        if word_list[key] > 2:
            new_word_list[key] = word_list[key]
    with open(bagfile, 'w') as f:
        f.write(json.dumps(new_word_list))
        f.close()


def get_data(infile):
    """
    Uses bag of words representation to create feature matrix X. Also returns output labels y.
    """
    raw_data = load(infile)
    comments, y = extract(raw_data)
    word_list = json.load(open('../data/bagfile_subset.json'))
    print len(word_list)
    n, d = len(comments), len(word_list)
    X = np.zeros((n, d))
    for i in range(len(comments)):
        words = extract_words(comments[i])
        for j in range(len(words)):
            if words[j] in word_list:
                X[i, j] = 1
    return np.asarray(X), np.asarray(y)


def main():
    create_and_write_dictionary('../data/features_data.csv')


if __name__ == '__main__':
    main()
