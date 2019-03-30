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

# Splitting the dataset into train and validate
df, validate_set = train_test_split(main_df, test_size=0.20, random_state=0)

# Create the dictionary in TFIDF
# There are too many unique words. Set max features to 5000
TFIDF_vect = TfidfVectorizer(max_features=5000)
TFIDF_vect.fit(main_df['processed_content'])

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

