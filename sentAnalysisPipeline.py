import nltk, re, csv,time
from gensim import corpora
# The following list is to further remove some frequent words in SGNews.
from sklearn.model_selection import train_test_split,ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import pickle
from textblob import TextBlob
from sentimentAnalysisUtil import stemmed_words,get_top_n_words
import pandas as pd
import statistics
from sklearn.pipeline import Pipeline

startTime = time.time()
stop_list = nltk.corpus.stopwords.words('english')

field='Content'
labelField='polarity'
docs=[]
label=[]

main_df = pd.read_csv("data/preprocessed_reviewinfo.csv")

#using textblob as a lexicon
"""
for i in range(0,5):
    print(TextBlob(docs[i]).sentiment)
    print(docs[i])"""

print('Finished reading sentences from the training data file. Time: ', time.time()-startTime )

#x_train, x_test, y_train, y_test = train_test_split(docs,label,test_size =0.3, random_state=50)
df, validate_set = train_test_split(main_df, test_size=0.20, random_state=0)
countVect = CountVectorizer(max_features=1000, lowercase=True, stop_words= 'english', ngram_range=(1,2))
tfidf = TfidfTransformer(use_idf=True, smooth_idf=True)
### Support Vector Machine
clf = svm.LinearSVC()
filename = 'model_sentiment/svm_model.pk'





# un comment model for fitting
'''
### Logistic Regression
logRegression = LogisticRegression()
filename = 'model_sentiment/logistic_regression_model.pk'
'''

'''
## Naive bayes 
clf = MultinomialNB()
filename = 'model_sentiment/nb_model.pk
'''

sentiment_pipeline = Pipeline(steps=[
        ('vect', countVect),
        ('tfidf', tfidf),
        ('classifier', clf)
    ])
"""
pipeline= Pipeline([('count',CountVectorizer(max_features=1000, lowercase=True, stop_words= 'english', ngram_range=(1,1),analyzer = stemmed_words)),
 ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=True)), ('clf', LogisticRegression())
])
"""
#Scores of average naive bayes classifier in cross validation
scores = []
count = 1

#Instantiate cross validation folds
ss = ShuffleSplit(n_splits=5, test_size=0.20, random_state=0)
counter =1
# cross validation,k = 5
for train_index, test_index in ss.split(df):
    train_df = df.iloc[train_index] #the 4 partitions
    test_df = df.iloc[test_index] #the 1 partition to test
    x_train, y_train = train_df['Content'].tolist(),train_df['polarity'].tolist()
    x_test, y_test= test_df['Content'].tolist(),test_df['polarity'].tolist()

    # Preparing documents into list according to categories
    start = time.time()
    sentiment_pipeline.fit(x_train,y_train)
    prediction=sentiment_pipeline.predict(x_test)
    print("Iteration " + str(counter) + " Model accuracy : " + str(np.mean(prediction==y_test)))
    counter=counter+1
    #add to list of scores    
    
    end = time.time()
    print("time taken: " + str((end - start)) + " secs")

print("\nCross Validation Average Score: " + str(statistics.mean(scores)))
print("Time taken: " + str(time.time() - startTime))
"""


prediction = model.predict(tfidf.transform(count.transform(x_test)))

print("Model accuracy : " + str(np.mean(prediction==y_test)))

pickle.dump(model, open(filename, 'wb'))
pickle.dump(tfidf, open('model_sentiment/tfidf_trans.pk', 'wb'))
pickle.dump(count, open('model_sentiment/count_vert.pk', 'wb'))


print('\nClasification report:\n', classification_report(y_test, prediction))
print('\nConfussion matrix:\n',confusion_matrix(y_test, prediction)  )
    


z_test = [input("What is your review? ")]
prediction = model.predict(tfidf.transform(count.transform(z_test)))
print("prediction is:" + prediction[0])"""