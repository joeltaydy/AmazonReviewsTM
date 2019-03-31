import nltk, re, csv,time
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from nltk.corpus import stopwords

from gensim import corpora

import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import pickle
import pandas as pd
import re
import statistics
import random

#Load the saved feature extraction 
classifier_saved = open("model_classification/FeatureExtraction.pickle", "rb") #binary read
classifier = pickle.load(classifier_saved)
classifier_saved.close()

#input
n = [input("Please input the top n features")]
# Extract the key features and put into dataframe

list_1 = classifier.most_informative_features(int(n))
df_important_features = pd.DataFrame(columns=['Feature','Category_1',
                                              'Category_0','Cat1_Cat0','Ratio','Ratio_1'])

for (fname, fval) in list_1:
    cpdist = classifier._feature_probdist
    
    def labelprob(l):
        return cpdist[l, fname].prob(fval)

    labels = sorted(
        [l for l in classifier._labels if fval in cpdist[l, fname].samples()],
        key=labelprob
    )
    
    if len(labels) == 1:
        continue
    l0 = labels[0]
    l1 = labels[-1]
    if cpdist[l0, fname].prob(fval) == 0:
        ratio = 'INF'
    else:
        ratio = round(cpdist[l1, fname].prob(fval) / cpdist[l0, fname].prob(fval), 1)
        fname = fname.replace('contains(','')
        fname = fname.replace(')','')        
        df_important_features.loc[len(df_important_features)] = [fname, l1, l0, l1+" : "+l0, 
                                                ratio, str(ratio)+" : 1.0"]
 
return df_important_features