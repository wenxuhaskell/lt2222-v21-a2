import sys
import os
import numpy as np
import numpy.random as npr
import pandas as pd
import random

# Module file with functions that you fill in so that they can be
# called by the notebook.  This file should be in the same
# directory/folder as the notebook when you test with the notebook.

# You can modify anything here, add "helper" functions, more imports,
# and so on, as long as the notebook runs and produces a meaningful
# result (though not necessarily a good classifier).  Document your
# code with reasonable comments.

# Function for Part 1
def preprocess(inputfile):
    return inputfile.readlines()

# Code for part 2
class Instance:
    def __init__(self, neclass, features):
        self.neclass = neclass
        self.features = features

    def __str__(self):
        return "Class: {} Features: {}".format(self.neclass, self.features)

    def __repr__(self):
        return str(self)

def create_instances(data):
    instances = []
    for i in range(100):
        neclass = random.choice(['art','eve','geo','gpe','nat','org','per','tim'])
        features = ["something"] * random.randint(1, 10)
        instances.append(Instance(neclass, features))
    return instances

# Code for part 3
def create_table(instances):
    df = pd.DataFrame()
    df['class'] = [random.choice(['art','eve','geo','gpe','nat','org','per','tim']) for i in range(100)]
    for i in range(3000):
        df[i] = npr.random(100)

    return df

def ttsplit(bigdf):
    df_train = pd.DataFrame()
    df_train['class'] = [random.choice(['art','eve','geo','gpe','nat','org','per','tim']) for i in range(80)]
    for i in range(3000):
        df_train[i] = npr.random(80)

    df_test = pd.DataFrame()
    df_test['class'] = [random.choice(['art','eve','geo','gpe','nat','org','per','tim']) for i in range(20)]
    for i in range(3000):
        df_test[i] = npr.random(20)
        
    return df_train.drop('class', axis=1).to_numpy(), df_train['class'], df_test.drop('class', axis=1).to_numpy(), df_test['class']

# Code for part 5
def confusion_matrix(truth, predictions):
    print("I'm confusing.")
    return "I'm confused."

# Code for bonus part B
def bonusb(filename):
    pass
