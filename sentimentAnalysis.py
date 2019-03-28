import nltk, re, csv,time
from gensim import corpora
# The following list is to further remove some frequent words in SGNews.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import pickle
from textblob import TextBlob
from sentimentAnalysisUtil import stemmed_words

startTime = time.time()
stop_list = nltk.corpus.stopwords.words('english')

field='Content'
labelField='polarity'
docs=[]
label=[]
with open("data/preprocessed_reviewinfo.csv",encoding='utf-8') as csvfile:
            sampleData = []
            reader = csv.DictReader(csvfile)
            counter =1
            for row in reader:
                sampleData.append(row)
            for row in sampleData:
                label.append(row[labelField])        
                docs.append(row[field])

#using textblob as a lexicon
"""
for i in range(0,5):
    print(TextBlob(docs[i]).sentiment)
    print(docs[i])"""

print('Finished reading sentences from the training data file. Time: ', time.time()-startTime )


print("finish converting to vector")
x_train, x_test, y_train, y_test = train_test_split(docs,label,test_size =0.3, random_state=50)



"""k_fold = KFold(n=len(x_train), n_folds=3)  
pipeline= Pipeline([('count',CountVectorizer(max_features=1000, lowercase=True, stop_words= 'english', ngram_range=(1,1),analyzer = stemmed_words)),
 ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=True)), ('clf', LogisticRegression())
])
"""

count = CountVectorizer(max_features=1000, lowercase=True, stop_words= 'english', ngram_range=(1,2),analyzer = stemmed_words)
x_train_temp = count.fit_transform(x_train)
tfidf = TfidfTransformer(use_idf=True, smooth_idf=True)
x_train_tfidf = tfidf.fit_transform(x_train_temp)


# un comment model for fitting

### Logistic Regression
logRegression = LogisticRegression()
model = logRegression.fit(x_train_tfidf,y_train)
filename = 'model_sentiment/logistic_regression_model.pk'


'''
## Naive bayes 
clf = MultinomialNB()
model= clf.fit(temp2,y_train)
filename = 'model_sentiment/nb_model.pk
'''

'''
### Support Vector Machine
clf = svm.LinearSVC()
model= clf.fit(temp2,y_train)
filename = 'model_sentiment/svm_model.pk'
'''

prediction = model.predict(tfidf.transform(count.transform(x_test)))

print("Model accuracy : " + str(np.mean(prediction==y_test)))

pickle.dump(model, open(filename, 'wb'))
pickle.dump(tfidf, open('model_sentiment/tfidf_trans.pk', 'wb'))
pickle.dump(count, open('model_sentiment/count_vect.pk', 'wb'))


print('\nClasification report:\n', classification_report(y_test, prediction))
print('\nConfusion matrix:\n',confusion_matrix(y_test, prediction)  )
    
print("Time taken: " + str(time.time() - startTime))

z_test = [input("What is your review? ")]
prediction = model.predict(tfidf.transform(count.transform(z_test)))
print("Prediction:" + prediction[0])