import sys
import os
import numpy as np
import numpy.random as npr
import pandas as pd
import random
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk import FreqDist

# Module file with functions that you fill in so that they can be
# called by the notebook.  This file should be in the same
# directory/folder as the notebook when you test with the notebook.

# You can modify anything here, add "helper" functions, more imports,
# and so on, as long as the notebook runs and produces a meaningful
# result (though not necessarily a good classifier).  Document your
# code with reasonable comments.

# stop words list
stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", 
                     "your", "yours", "yourself", "yourselves", "he", "him", "his", 
                     "himself", "she", "her", "hers", "herself", "it", "its", "itself", 
                     "they", "them", "their", "theirs", "themselves", "what", "which", 
                     "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
                     "was", "were", "be", "been", "being", "have", "has", "had", "having", 
                     "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", 
                     "or", "because", "as", "until", "while", "of", "at", "by", "for", 
                     "with", "about", "against", "between", "into", "through", "during", 
                     "before", "after", "above", "below", "to", "from", "up", "down", "in", 
                     "out", "on", "off", "over", "under", "again", "further", "then", "once", 
                     "here", "there", "when", "where", "why", "how", "all", "any", "both", 
                     "each", "few", "more", "most", "other", "some", "such", "no", "nor", 
                     "not", "only", "own", "same", "so", "than", "too", "very", "'s", "t", 
                     "can", "will", "just", "don't", "didn't", "doesn't", "hadn't", "hasn't", 
                     "haven't", "aren't", "isn't", "can't", "could", "couldn't", "should", 
                     "shouldn't", "shall", "shan't", "wouldn't", "won't", "now"]

# '$' for padding before NE
# '&' for padding after NE
padding_symbols = ['$', '&']

# NE types
entity_types = ['art', 'eve', 'geo', 'gpe', 'nat', 'org', 'per', 'tim']

# size of context
cxt_size = 5

# Function for Part 1
def preprocess(inputfile):
    lines = inputfile.readlines()
    rows = list(map( lambda x: x.rstrip('\n').split("\t"), lines))
    # conversion to lower case and remove punktuations
    r =list()
    for s in rows:
        s1 = [w.lower() for w in s]
        if (s1[2] not in stopwords) and s1[2] != '""""':
           r.append(s1)

    return r[1:]

# Code for part 2
class Instance:
    def __init__(self, neclass, features):
        self.neclass = neclass
        self.features = features

    def __str__(self):
        return "Class: {} Features: {}".format(self.neclass, self.features)

    def __repr__(self):
        return str(self)

    def getNEClass(self):
        return self.neclass

    def getNEFeatures(self):
        return self.features
    
    def toTuple(self):
        return (self.neclass, self.features)

# Code for part 2
def create_instances(data, pos=False):
    instances = []
    data_size = len(data)
    count = 0
    while count < data_size-1:
        if data[count][4][0] == 'b':
            t_count = count
            # initialize context
            features = []
            # entity type
            neclass = data[t_count][4][2:]
            # find context before NE
            if t_count > 0:
                m = 0
                if t_count > 5:
                    m = t_count - 5
                n = t_count
                while n > m:
                    n = n-1
                    if data[n][2] == '.':
                        break
                    elif data[n][2] == ',':
                        if m > 0:
                            m = m - 1
                        continue
                    else:
                        features.insert(0, data[n][2])
                        if pos == True:
                            features.insert(0, data[n][3])
            
            if pos == True:
                n = cxt_size*2-len(features)
            else:
                n = cxt_size-len(features)
                
            if n > 0:
                padding_s = [padding_symbols[0]]*n
                features = padding_s + features
            
            # skip 'I-xyz'
            while t_count < data_size-1:
                t_count = t_count + 1
                if data[t_count][4][0] != 'i':
                     break

            # find context after NE
            m = t_count + cxt_size - 1
            if m > data_size:
                m = data_size
            
            s_count = t_count
            while s_count < m:
                if data[s_count][2] == '.':
                    break
                else:
                    if data[s_count][2] != ',' and data[s_count][4][0] != 'i':
                        features.append(data[s_count][2])
                        if pos == True:
                            features.append(data[s_count][3])
                    else:
                        # ignore ','
                        if m < data_size:
                            m = m + 1
                    s_count = s_count + 1

            if pos == True:
                m = cxt_size*4 - len(features)
            else:
                m = cxt_size - len(features)
            if m > 0:
                padding_e = [padding_symbols[1]]*m
                features.extend(padding_e)

            # add an instance
            if features != []:
                instances.append(Instance(neclass, features))

            # continue while loop 
            count = t_count
        else:
            count = count + 1

    return instances


# Code for part 3
def create_table(instances):
    type_list = [instance.getNEClass() for instance in instances]
    text_list = [instance.getNEFeatures() for instance in instances]
        
    all_tokens = [x for list in text_list for x in list]
    all_counts = FreqDist(all_tokens)
    
    top_freqwords = all_counts.most_common(3000)
    sentence_borders = ['$', '&']
    filtered_freqwords = [(word, freq) for word, freq in top_freqwords if word not in sentence_borders]
    
    dict_list = [x for (x, y) in filtered_freqwords]
    index_list = np.arange(0,len(dict_list),1).tolist()

    # unpack the values from the zip object
    listz = [*zip(dict_list, index_list)]

    # create a dictionary of frequent words
    # A key:value is frequent word:index
    # In our assignment, it will look like 'the':0, 'and':1, 'to':2 and so on.
    dict_t = dict(listz)

    freqwords_set = set(dict_list)

    freq_tt = []
    netypes = []
    # Iterate each sentence (text) in the list
    for text, cls in zip(text_list, type_list):
        # fill it up with all '0's
        n_tt = [0]*len(freqwords_set)
        # iterate over each word in the sentence
        modified = False
        for e in text:
            try :
                # if the word is in the dictory of frequent words, 
                # we will get its index in all columns.
                i = dict_t[e]
                # The corresponding entry in the sentence will be marked
                n_tt[i] = n_tt[i] + 1
                modified = True
            except (KeyError) :
                # do nothing
                i = 0
        # if the sentence contains any of frequent words
        if modified == True:
            freq_tt.append(n_tt)
            netypes.append(cls)


    freq_tt_trans = np.transpose(np.matrix(freq_tt))

    freq_tt_m = np.matrix(freq_tt)
    total_documents = len(freq_tt)
    
    # array of total terms (per row)
    total_terms_per_row_arr = np.sum(freq_tt, axis=1)
    # array of number of documents containing a given term
    documents_with_term_arr = np.count_nonzero(freq_tt, axis=0)
    
    # Array of Inverse Document Frequency of terms
    idf_arr = np.log(total_documents/documents_with_term_arr)
        
    tfidf_list = []
    for r, total_terms in zip(freq_tt, total_terms_per_row_arr):
        # for each row, calculate tf x idf
        tf_idf = np.array(r*idf_arr/total_terms, copy=True)
        tfidf_list.append(tf_idf.tolist())

    tfidfm = np.asarray(tfidf_list)
    
    tfidfm_trans = np.transpose(tfidfm)
    # create a DataFrame with the first column including file name
    df = pd.DataFrame({'class' : netypes})
    for w, v in zip(dict_list, tfidfm_trans):
        df[w] = v

    return df

def ttsplit(bigdf):
    
    size = len(bigdf)
    all_idx = range(len(bigdf))
    size_test = int(round(size/5))
    size_train = size - size_test
    
    test_idx = random.sample(all_idx, size_test)
    train_idx = [x for x in all_idx if x not in test_idx]
    
    df_test = bigdf.iloc[test_idx, :]
    df_train = bigdf.iloc[train_idx, :]

    return df_train.drop('class', axis=1).to_numpy(), df_train['class'], df_test.drop('class', axis=1).to_numpy(), df_test['class']

# Code for part 5
def confusion_matrix(truth, predictions):
    # create a dictionary to map the index in array to entity types (columns and indexes)
    idx_dict = dict([*zip(entity_types, range(len(entity_types)))])
    t_values = truth.tolist()   
    #print(predictions)
    p_values = predictions.tolist()

    m = np.zeros(shape=(len(entity_types),len(entity_types)))
    for t, p in zip(t_values, p_values):
        i_t = idx_dict[t]
        i_p = idx_dict[p]
        
        m[i_t, i_p] = m[i_t, i_p] + 1

    df = pd.DataFrame(data=m, index=entity_types, columns=entity_types)
    
    sum_per_row = np.sum(m, axis=1)
    accuracy = [] 
    for i, r in zip(range(len(entity_types)), sum_per_row):
        if r != 0:
            accuracy.append(m[i,i]/r)
        else:
            accuracy.append(0)
    
    df['Accuracy'] = accuracy
    
    return df

# Code for bonus part B
def bonusb(filename):
    # preprocess data
    gmbfile = open(filename, "r")
    inputdata = preprocess(gmbfile)
    gmbfile.close()
    
    # create instances
    instances = create_instances(inputdata, pos=True)
    
    # create table
    bigdf = create_table(instances)
    
    # split into train set and test set
    train_X, train_y, test_X, test_y = ttsplit(bigdf)

    # train the model
    model = LinearSVC()
    model.fit(train_X, train_y)
    train_predictions = model.predict(train_X)
    test_predictions = model.predict(test_X)
    
    # confusion matrix for evaluating the performance of the trained model
    cm1 = confusion_matrix(test_y, test_predictions)
    print("Confusion matrix for test set")
    print(cm1)
    
    cm2 = confusion_matrix(train_y, train_predictions)
    print("Confusion matrix for train set")
    print(cm2)
    
    return cm1, cm2
