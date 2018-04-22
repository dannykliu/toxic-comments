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

import util
import csv


def get_indices(infile):
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
    return data, header, toxic_indices, nontoxic_indices


def write_data(outfile, data, header, toxic_indices, nontoxic_indices, num_toxic, num_nontoxic):
    random_toxic_indices = np.random.choice(np.arange(len(toxic_indices)), size=num_toxic, replace=False)
    random_nontoxic_indices = np.random.choice(np.arange(len(nontoxic_indices)), size=num_nontoxic, replace=False)
    toxic_indices = toxic_indices[random_toxic_indices]
    nontoxic_indices = nontoxic_indices[random_nontoxic_indices]

    with open(outfile, 'wb') as csvOut:
        out = csv.writer(csvOut, delimiter=',')
        out.writerow(header)
        for i in nontoxic_indices:
            out.writerow(data[i])
        for i in toxic_indices:
            out.writerow(data[i])


def subset_data(infile, outfile):
    """
    Create subset of 20% of original dataset, with same class proportions as original dataset
    """
    data, header, toxic_indices, nontoxic_indices = get_indices(infile)
    ratio = 0.15
    num_toxic = int(ratio * len(toxic_indices))
    num_nontoxic = int(ratio * len(nontoxic_indices))
    print num_toxic, num_nontoxic
    write_data(outfile, data, header, toxic_indices, nontoxic_indices, num_toxic, num_nontoxic)


def subsample_data(infile, outfile, n):
    """
    Subsample dataset with n examples of positive class and n examples of negative class
    """
    data, header, toxic_indices, nontoxic_indices = get_indices(infile)
    write_data(outfile, data, header, toxic_indices, nontoxic_indices, n, n)


subset_data('../data/train.csv', '../data/subset.csv')
# util.create_and_write_dictionary('../data/subset.csv', '../data/bagfile_subset.json')

# subsample_data('../data/train.csv', '../data/subsample.csv', 2000)
# util.create_and_write_dictionary('../data/subsample.csv', '../data/bagfile_subsample.json')
