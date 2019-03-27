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
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.model_selection import ShuffleSplit
import pickle
from textblob import TextBlob

startTime = time.time()
stop_list = nltk.corpus.stopwords.words('english')
stemmer = nltk.stem.porter.PorterStemmer()

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

analyzer = CountVectorizer().build_analyzer()
def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

k_fold = KFold(n_folds=3)  
"""
pipeline= Pipeline([('count',CountVectorizer(max_features=1000, lowercase=True, stop_words= 'english', ngram_range=(1,1),analyzer = stemmed_words)),
 ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=True)), ('clf', LogisticRegression())
])
"""

### Logistic Regression
logRegression = LogisticRegression()

filename = 'model_sentiment/logistic_regression_model.pk'

score=[]

for train_index, test_index in k_fold.split(x_train):
    print(train_index)
    print(test_index)
    print()

    count = CountVectorizer(max_features=1000, lowercase=True, stop_words= 'english', ngram_range=(1,1),analyzer = stemmed_words)
    x_train2,x_test2,y_train2,y_test = x_train[train_index],x_train[test_index],y_train[train_index],y_train[test_index]
    temp = count.fit_transform(x_train2)
    tfidf = TfidfTransformer(use_idf=True, smooth_idf=True)
    temp2 = tfidf.fit_transform(temp)
    model = logRegression.fit(temp2,y_train2)
    prediction = model.predict(tfidf.transform(count.transform(x_test2)))
    score.append(np.mean(prediction==y_test))
    print("Model accuracy : " + str(np.mean(prediction==y_test)))


print( sum(score)/len)

# un comment model for fitting

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

#prediction = model.predict(tfidf.transform(count.transform(x_test)))

#print("Model accuracy : " + str(np.mean(prediction==y_test)))

#pickle.dump(model, open(filename, 'wb'))
#pickle.dump(tfidf, open('model_sentiment/tfidf_trans.pk', 'wb'))
#pickle.dump(count, open('model_sentiment/count_vert.pk', 'wb'))


#print('\nClasification report:\n', classification_report(y_test, prediction))
#print('\nConfussion matrix:\n',confusion_matrix(y_test, prediction)  )
    
#print("Time taken: " + str(time.time() - startTime))

#z_test = [input("What is your review? ")]
#prediction = model.predict(tfidf.transform(count.transform(z_test)))
#print("prediction is:" + prediction[0])