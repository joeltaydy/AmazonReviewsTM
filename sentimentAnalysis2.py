from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import numpy as np

filename = 'model_sentiment/logistic_regression_model.pk'
#filename = 'model_sentiment/nb_model.pk
#filename = 'model_sentiment/svm_model.pk'

tfidf = TfidfTransformer()
count = CountVectorizer()

# load the model from disk
model = pickle.load(open(filename, 'rb'))
x_test = ["this is wonderful"]
prediction = model.predict(tfidf.transform(count.transform(x_test)))

print(prediction)