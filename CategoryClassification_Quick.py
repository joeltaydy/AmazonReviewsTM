import sklearn
from sklearn.model_selection import train_test_split,ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm

import numpy as np
from sklearn import svm
import pickle
import pandas as pd


# Load the processed reviews
main_df = pd.read_csv('data/processed_reviews.csv')

# Load the classifier
classifier_saved = open("model_classification/CategoryClassifier.pickle", "rb")
classifier = pickle.load(classifier_saved)
classifier_saved.close()

#Load the saved classifier 
classifier_saved = open("model_classification/TFIDF_Reviews_Category.pickle", "rb") #binary read
TFIDF_vect = pickle.load(classifier_saved)
classifier_saved.close()

#input
z_test = [input("What is your review? ")]

#process input
review_holder = np.array([z_test])
review_holder_1 = pd.Series(review_holder)

# Transform reviews into TFIDF
review_TFIDF = TFIDF_vect.transform(review_holder_1)
prediction = classifier.predict(test_TDIDF)

if prediction[0] == 0:
    print("Review category: Camera")
elif prediction[0] == 1:
    print("Review category: Laptop")
else
    print("Review category: Mobile Phone")

