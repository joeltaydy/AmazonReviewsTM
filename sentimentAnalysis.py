import nltk, re, csv,time
from gensim import corpora
from random import shuffle
stop_list = nltk.corpus.stopwords.words('english')
# The following list is to further remove some frequent words in SGNews.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.metrics import classification_report,confusion_matrix
import pickle
startTime = time.time()

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
            shuffle(sampleData) # Shuffle the data around
            for row in sampleData:
                label.append(row[labelField])        
                docs.append(row[field])


print('Finished reading sentences from the training data file. Time: ', time.time()-startTime )


print("finish converting to vector")
x_train, x_test, y_train, y_test = train_test_split(docs,label,test_size =0.2)
count = CountVectorizer()
temp = count.fit_transform(x_train)

tfidf = TfidfTransformer()
temp2 = tfidf.fit_transform(temp)


# un comment model for fitting

### Logistic Regression
logRegression = LogisticRegression()
model = logRegression.fit(temp2,y_train)
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
print(type(x_test))
print(x_test[:3])
prediction = model.predict(tfidf.transform(count.transform(x_test)))
print(prediction[:5])
print(y_test[:5])
print("Model accuracy : " + str(np.mean(prediction==y_test)))

pickle.dump(model, open(filename, 'wb'))

print('\nClasification report:\n', classification_report(y_test, prediction))
print('\nConfussion matrix:\n',confusion_matrix(y_test, prediction)  )
    
print("Time taken: " + str(time.time() - startTime))