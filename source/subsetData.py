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


def load(infile, predict_cols = [2,3,4,5,6,7]):
    """
    """
    index = 0

    data = []   # list of lists, where each sublist is a line of the csv file
    toxic_indices = []  # indices in data (lines of csv file) where comment contains SOME level of toxicity
    nontoxic_indices = []  # indices in data where comment is completely non-toxic

    with open(infile, 'rU') as fid:
        csvData = csv.reader(fid, delimiter=',')
        data =[]

        index= 0
        for line in csvData:
            data.append(line)
            if index!=0:
                if any([int(line[predict_col]) == 1 for predict_col in predict_cols]):
                    toxic_indices.append(index)
                else: 
                    nontoxic_indices.append(index)
            else: 
                header = line
            index+=1
        
    with open('subsampled_train.csv', 'wb') as csvOut:
        out = csv.writer(csvOut, delimiter=',')
        out.writerow(header)

        #write non-toxic comments to sub-sampled file
        for i in range(len(toxic_indices)):
            rand_index = np.random.randint(1, len(nontoxic_indices)) #generate random index
            nontoxic_sample_index = nontoxic_indices[rand_index]
            out.writerow(data[nontoxic_sample_index])

            #sample without replacement so remove the index we used
            nontoxic_indices.remove(nontoxic_sample_index)
        # write all toxic comments to sub-sampled file
        for i in toxic_indices:
            out.writerow(data[i])
        

load('train.csv') 
