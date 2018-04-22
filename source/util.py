import numpy as np
import csv
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


def extract(data):
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
        x.append(line[comment_col])
    return x, y


def clean_text(text):
    """ Removes punctuation, capitalizations, numbers, stop words, and stems words"""
    ps = PorterStemmer()

    stop_words = set(stopwords.words('english'))
  
    text = text.lower()
    text = contractions.expandContractions(text)
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
    text = re.sub(r'(.)\1\1+', r'\1\1', text) # letters repeated 3 or more times in a row are repeated twice
    text= re.sub(r'(ha)\1\1+', r'haha', text) 
    text= re.sub(r'(lo)\1\1+', r'lol', text)
    text = text.strip(' ')

    # stem words
    tokenizer = WhitespaceTokenizer()
    tokenized_comment = tokenizer.tokenize(text) 
    filtered_sentence = [w for w in tokenized_comment if not w in stop_words]
    stemmed_comment = [ps.stem(word) for word in filtered_sentence]
    text = " ".join(stemmed_comment)
    return text


def get_data(infile, clean=True):
    raw_data = load(infile)
    comments, y = extract(raw_data)
    raw = np.array(comments)
    if clean:
        for i in range(len(comments)):
            comments[i] = clean_text(comments[i])
    return np.array(comments), np.array(y), raw


def get_features(comments):
    cap_per = get_cap_percentage(comments)
    p_ind1, p_ind2 = get_period_indicators(comments)
    ex_ind1, ex_ind2, ex_ind3, ex_ind4, ex_ind5 = get_exclamation_indicators(comments)
    features = [cap_per, p_ind1, p_ind2, ex_ind1, ex_ind2, ex_ind3, ex_ind4, ex_ind4, ex_ind5]
    return np.array(features)


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


def get_cap_percentage(X):
    """
    Gets the capitalization percentage for each comment.
    """
    cap_per = []
    for comment in X:
        stripped_comment = comment.replace(" ", "")
        if len(stripped_comment) != 0:
            uppers = [l for l in stripped_comment if l.isupper()]
            percentage = 1.0*len(uppers)/len(stripped_comment)
            cap_per.append(percentage)
        else:
            cap_per.append(0)
    return np.array(cap_per)


def get_exclamation_indicators(X):
    """
    Gets the exclamation point percentage for each comment.
    """
    no_ex = []
    one_ex = []
    two_ex = []
    three_four_ex = []
    five_plus_ex = []
    for comment in X:
        count = comment.count('!')
        if count == 0:
            no_ex.append(1)
        else:
            no_ex.append(0)
        if count == 1:
            one_ex.append(1)
        else:
            one_ex.append(0)
        if count == 2:
            two_ex.append(1)
        else:
            two_ex.append(0)
        if count == 3 or count == 4:
            three_four_ex.append(1)
        else:
            three_four_ex.append(0)
        if count >= 5:
            five_plus_ex.append(1)
        else:
            five_plus_ex.append(0)
    return np.array(no_ex), np.array(one_ex), np.array(two_ex), np.array(three_four_ex), np.array(five_plus_ex)


def get_period_indicators(X):
    no_periods = []
    two_periods = []
    for comment in X:
        count = comment.count('.')
        if count == 0:
            no_periods.append(1)
        else:
            no_periods.append(0)
        if count >= 1:
            two_periods.append(1)
        else:
            two_periods.append(0)
    return np.array(no_periods), np.array(two_periods)


def main():
    pass


if __name__ == '__main__':
    main()
