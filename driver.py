import numpy as np
import csv

'''
Your status report should describe what you have accomplished so far and what else you plan to do. You should use this report to help you make sure you are on-track and to provide staff with an update on your current progress.
For example, if you are doing an application project, then by this point, you should have tried some initial visualizations, defined the features / subsets of your data, tried out at least one machine learning algorithm, and visualized and summarized the results.
An example report is provided at the end of this document. This is not meant to take you a long time, but please do spend a little bit of effort putting this together. Please keep in mind that the intended audience is the course staff.

Sasha & Amelia: Visualizations
Dalton & Danny: Features, train/test split, model
'''

with open('train.csv', 'r') as file:
    for line in file:
        print(line)