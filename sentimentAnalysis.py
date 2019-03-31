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
from sentimentAnalysisUtil import stemmed_words,get_top_n_words,removeStopwords
import pandas as pd
import statistics


startTime = time.time()
field='Content'
labelField='polarity'
docs=[]
label=[]

main_df = pd.read_csv("data/preprocessed_reviewinfo.csv")

#print((main_df))
"""with open("data/preprocessed_reviewinfo.csv",encoding='utf-8') as csvfile:
            sampleData = []
            reader = csv.DictReader(csvfile)
            counter =1
            for row in reader:
                sampleData.append(row)
            for row in sampleData:
                label.append(row[labelField])        
                docs.append(row[field])
"""
#using textblob as a lexicon
"""
for i in range(0,5):
    print(TextBlob(docs[i]).sentiment)
    print(docs[i])"""

print('Finished reading sentences from the training data file. Time: ', time.time()-startTime )

#x_train, x_test, y_train, y_test = train_test_split(docs,label,test_size =0.3, random_state=50)
df, validate_set = train_test_split(main_df, test_size=0.20, random_state=0)
positive_df = df[df.polarity == 1]
negative_df = df[df.polarity ==0]
difference = positive_df/negative_df
df = pd.concat([negative_df, negative_df,negative_df,negative_df,positive_df])

"""k_fold = KFold(n=len(x_train), n_folds=3)  
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
    x_train, y_train = removeStopwords(train_df['Content'].tolist()), train_df['polarity'].tolist()
    x_test, y_test= removeStopwords(test_df['Content'].tolist()),test_df['polarity'].tolist()

    # Preparing documents into list according to categories
    start = time.time()
    count = CountVectorizer(max_features=5000, lowercase=True, ngram_range=(1,2),analyzer = stemmed_words)
    temp = count.fit_transform(x_train)
    tfidf = TfidfTransformer(use_idf=True, smooth_idf=True)
    temp2 = tfidf.fit_transform(temp)

    
    """### Logistic Regression
    logRegression = LogisticRegression()
    model = logRegression.fit(temp2,y_train)
    filename = 'model_sentiment/logistic_regression_model.pk'"""

    
    """## Naive bayes 
    clf = MultinomialNB()
    model= clf.fit(temp2,y_train)
    filename = 'model_sentiment/nb_model.pk'
"""
    
    ### Support Vector Machine
    clf = svm.LinearSVC()
    model= clf.fit(temp2,y_train)
    filename = 'model_sentiment/svm_model.pk'
    
    prediction = model.predict(tfidf.transform(count.transform(x_test)))

    print("Iteration " + str(counter) + " Model accuracy : " + str(np.mean(prediction==y_test)))
    counter=counter+1
    #add to list of scores
    scores.append(np.mean(prediction==y_test))
    #get_top_n_words(temp,count)
    weights = np.asarray(temp2.mean(axis=0)).ravel().tolist()
    weights_df = pd.DataFrame({'term': count.get_feature_names(), 'weight': weights})
    print(weights_df.sort_values(by='weight', ascending=False).head(20))    
    end = time.time()
    print("time taken: " + str((end - start)) + " secs")

print("\nCross Validation Average Score: " + str(statistics.mean(scores)))
print("Time taken: " + str(time.time() - startTime))


print("*"*10+ "Training final model" + "*"*10 )
x_train, y_train = removeStopwords(df['Content'].tolist()), df['polarity'].tolist()
x_test, y_test= removeStopwords(validate_set['Content'].tolist()),validate_set['polarity'].tolist()

count = CountVectorizer(max_features=5000, lowercase=True, ngram_range=(1,2),analyzer = stemmed_words)
temp = count.fit_transform(x_train)
tfidf = TfidfTransformer(use_idf=True, smooth_idf=True)
temp2 = tfidf.fit_transform(temp)


# un comment model for fitting
'''
### Logistic Regression
logRegression = LogisticRegression()
model = logRegression.fit(temp2,y_train)
filename = 'model_sentiment/logistic_regression_model.pk'
'''

'''
## Naive bayes 
clf = MultinomialNB()
model= clf.fit(temp2,y_train)
filename = 'model_sentiment/nb_model.pk
'''


### Support Vector Machine
clf = svm.LinearSVC()
model= clf.fit(temp2,y_train)
filename = 'model_sentiment/svm_model.pk'


prediction = model.predict(tfidf.transform(count.transform(x_test)))

print("Model accuracy : " + str(np.mean(prediction==y_test)))

print("*"*10+ "Saving final model" + "*"*10 )
pickle.dump(model, open(filename, 'wb'))
pickle.dump(tfidf, open('model_sentiment/tfidf_trans.pk', 'wb'))
pickle.dump(count, open('model_sentiment/count_vert.pk', 'wb'))


print('\nClasification report:\n', classification_report(y_test, prediction))
print('\nConfussion matrix:\n',confusion_matrix(y_test, prediction)  )
    
print("time taken: " + str((time.time() - startTime)) + " secs")


z_test = [input("What is your review? ")]
prediction = model.predict(tfidf.transform(count.transform(z_test)))
print("prediction is:" + str(prediction[0]))