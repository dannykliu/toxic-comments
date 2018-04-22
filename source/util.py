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
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import contractions

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


def extract(data, clean=True) :
    """
    Parameters
    --------------------
        x      -- list of lists where each sublist is a line from a csv file

    Returns
    --------------------
        x      -- list of strings, length n (n comments)
        y      -- list of ints, 1 = toxic, 0 = non-toxic, labels for x
    """
    comment_col = 1
    predict_cols = [2, 3, 4, 5, 6, 7]
    x = []
    y = []

    for line in data:
        if any([int(line[predict_col]) == 1 for predict_col in predict_cols]):
            y.append(1)
        else: 
            y.append(0)
        if clean==True: 
            x.append(clean_text(line[comment_col]))
        else: 
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


def clean_text(text):
    """ Removes punctuation, capitalizations, numbers, stop words, and stems words"""
    ps = PorterStemmer()

    stop_words = set(stopwords.words('english'))
  
    text=text.lower()
    text=contractions.expandContractions(text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)  # remove punctuation
    text = re.sub('\s+', ' ', text)
    text = re.sub('\d+', ' ', text) # remove numbers
    text = text.strip(' ')

    # stem words
    tokenizer = WhitespaceTokenizer()
    tokenized_comment = tokenizer.tokenize(text) 
    filtered_sentence = [w for w in tokenized_comment if not w in stop_words]
    stemmed_comment = [ps.stem(word) for word in filtered_sentence]
    text = " ".join(stemmed_comment)
    return text


def create_and_write_dictionary(datafile, bagfile):
    """
    Create dictionary from all words in training data and write to text file
    """
    raw_data = load(datafile)
    comments, y = extract(raw_data)
    word_list = defaultdict(int)
    for comment in comments:
        words = extract_words(clean_text(comment))
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


def get_data2(infile, clean=True):
    raw_data = load(infile)
    comments, y = extract(raw_data, clean)
    return np.asarray(comments), np.asarray(y)


def get_data(infile):
    """
    Uses bag of words representation to create feature matrix X. Also returns output labels y.
    """
    raw_data = load(infile)
    comments, y = extract(raw_data)
    word_list = json.load(open('../data/bagfile_subset.json'))
    word_to_index = {}
    i = 0
    for word in word_list:
        word_to_index[word] = i
        i += 1
    n, d = len(comments), len(word_list)
    X = np.zeros((n, d))
    for i in range(len(comments)):
        words = extract_words(comments[i])
        for j in range(len(words)):
            if words[j] in word_list:
                X[i, word_to_index[words[j]]] = 1
    return np.asarray(X), np.asarray(y)


def entropy(y):
    probs = [np.mean(y == c) for c in set(y)]
    return np.sum(-p * np.log2(p) for p in probs)


def info_gain(Xj, y, threshold):
    less_than = np.where(Xj <= threshold)[0]
    greater_than = np.where(Xj > threshold)[0]
    prob_less = len(less_than) / float(len(Xj))
    H_less = entropy(y[less_than])
    H_greater = entropy(y[greater_than])
    cond_H = prob_less * H_less + (1-prob_less) * H_greater
    return entropy(y) - cond_H


def get_cap_percentage(X, y):
    """
    Gets the capitalization percentage for each comment.
    """
    cap_per = []
    for comment in X:
        stripped_comment = comment.replace(" ", "")
        if len(stripped_comment) != 0:
            cap_count = 0
            uppers = [l for l in stripped_comment if l.isupper()]
            percentage = 1.0*len(uppers)/len(stripped_comment)
            cap_per.append(percentage)
        else:
            cap_per.append(0)
    np_cap = np.array(cap_per)
    return np_cap


def get_exclamation_percentage(X, y):
    """
    Gets the exclamation point percentage for each comment.
    """
    ex_per = []
    for comment in X:
        stripped_comment = comment.replace(" ", "")
        if len(stripped_comment) != 0:
            ex_count = 0
            excl = [l for l in stripped_comment if l=="!"]
            percentage = 1.0*len(excl)/len(stripped_comment)
            ex_per.append(percentage)
        else:
            ex_per.append(0)
    np_ex = np.array(ex_per)
    return np_ex


def main():
    pass


if __name__ == '__main__':
    main()
