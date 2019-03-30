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
print(classifier.show_most_informative_features(n))