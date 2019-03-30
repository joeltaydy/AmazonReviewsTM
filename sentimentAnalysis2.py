from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import numpy as np
from sentimentAnalysisUtil import stemmed_words,removeStopwords

#filename = 'model_sentiment/logistic_regression_model.pk'
#filename = 'model_sentiment/nb_model.pk
filename = 'model_sentiment/svm_model.pk'
tfidf = pickle.load(open('model_sentiment/tfidf_trans.pk','rb'))
count = pickle.load(open('model_sentiment/count_vert.pk','rb'))
# load the model from disk
model = pickle.load(open(filename, 'rb'))

z_test = [input("What is your review? ")]
prediction = model.predict(tfidf.transform(count.transform(removeStopwords(z_test))))

print("prediction is:" + "positive rating" if prediction[0] == "1" else "prediction is: negative rating",)