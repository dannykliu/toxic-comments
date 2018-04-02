"""
Author: Sasha Friedrich
Date: 4/1/18
Description: Reads in a csv file of toxic comments. Creates a subsampled data set that has an equal number of toxic and non-toxic comments.
Non-toxic comments are sampled randomly without replacement. 
Note that in the output file ALL non-toxic comments are before toxic comments
outputfile: 'subsampled_train.csv'
"""


# numpy libraries
import numpy as np

import csv


def write_data(infile):
    """
    """
    predict_cols = [2, 3, 4, 5, 6, 7]
    data = []   # list of lists, where each sublist is a line of the csv file
    toxic_indices = []  # indices in data (lines of csv file) where comment contains SOME level of toxicity
    nontoxic_indices = []  # indices in data where comment is completely non-toxic
    header = None

    with open(infile, 'rU') as fid:
        csvData = csv.reader(fid, delimiter=',')
        index = 0
        for line in csvData:
            data.append(line)
            if index != 0:
                if any([int(line[predict_col]) == 1 for predict_col in predict_cols]):
                    toxic_indices.append(index)
                else: 
                    nontoxic_indices.append(index)
            else: 
                header = line
            index += 1

    toxic_indices = np.asarray(toxic_indices)
    nontoxic_indices = np.asarray(nontoxic_indices)
    random_toxic_indices = np.random.choice(np.arange(len(toxic_indices)), size=1000, replace=False)
    random_nontoxic_indices = np.random.choice(np.arange(len(nontoxic_indices)), size=1000, replace=False)
    toxic_indices = toxic_indices[random_toxic_indices]
    nontoxic_indices = nontoxic_indices[random_nontoxic_indices]

    with open('../data/subsampled_train_smaller.csv', 'wb') as csvOut:
        out = csv.writer(csvOut, delimiter=',')
        out.writerow(header)
        for i in nontoxic_indices:
            out.writerow(data[i])
        for i in toxic_indices:
            out.writerow(data[i])

        

write_data('../data/train.csv')
