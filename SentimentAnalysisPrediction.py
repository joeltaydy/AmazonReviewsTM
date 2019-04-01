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
from sentimentAnalysisUtil import stemmed_words,get_top_n_words,removeStopwords,preprocess_punc_stop
import pandas as pd
import statistics

'''
All model declaration
'''
all_model = [{'model_name':"logistic regression", "model": LogisticRegression()},{'model_name':"Naive Bayes", "model": MultinomialNB()},{'model_name':"SVM Linear", "model": svm.LinearSVC()} ]

startTime = time.time()
field='Content'
labelField='polarity'
docs=[]
label=[]

main_df = pd.read_csv("data/preprocessed_reviewinfo.csv")

print('Finished reading sentences from the training data file. Time: ', time.time()-startTime )

#x_train, x_test, y_train, y_test = train_test_split(docs,label,test_size =0.3, random_state=50)
df, validate_set = train_test_split(main_df, test_size=0.20, random_state=0)
positive_df = df[df.polarity == 1]
negative_df = df[df.polarity ==0]
difference = positive_df/negative_df
df = pd.concat([negative_df, negative_df,negative_df,negative_df,positive_df])

# df.assign(Content=removeStopwords(df['Content'].tolist()))
df.assign(Content=preprocess_punc_stop(df['Content'].tolist()))
print("pre processing donez")
#Scores of average Linear SVC in cross validation
scores = []
count = 1
#Instantiate cross validation folds
ss = ShuffleSplit(n_splits=5, test_size=0.20, random_state=0)
counter =1

for eachModel in all_model:
    print("*"*10 + "Model name: " + eachModel['model_name']+" "+"*"*10)
    # cross validation,k = 5
    startModelTime= time.time()
    for train_index, test_index in ss.split(df):
        train_df = df.iloc[train_index] #the 4 partitions
        test_df = df.iloc[test_index] #the 1 partition to test
        x_train, y_train = train_df['Content'].tolist(), train_df['polarity'].tolist()
        x_test, y_test=test_df['Content'].tolist(),test_df['polarity'].tolist()
        # x_train, y_train = preprocess_punc_stop(train_df['Content'].tolist()), train_df['polarity'].tolist()
        # x_test, y_test= preprocess_punc_stop(test_df['Content'].tolist()),test_df['polarity'].tolist()

        # print(x_train)
        # exit() 
        print("Converting to tfidf for " +eachModel['model_name'] )
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
        clf = eachModel['model']
        model= clf.fit(temp2,y_train)
        
        prediction = model.predict(tfidf.transform(count.transform(x_test)))

        print("Iteration " + str(counter) +" " + eachModel['model_name'] +" Model accuracy : " + str(np.mean(prediction==y_test)))
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
    print("Time taken: " + str(time.time() - startModelTime)) 

    print("*"*10+ "Training final model " +eachModel['model_name']+" " + "*"*10 )
    # x_train, y_train = df['Content'].tolist(), df['polarity'].tolist()
    # x_test, y_test= removeStopwords(validate_set['Content'].tolist()),validate_set['polarity'].tolist()
    x_train, y_train =df['Content'].tolist(), df['polarity'].tolist()
    x_test, y_test= preprocess_punc_stop(validate_set['Content'].tolist()),validate_set['polarity'].tolist()

    count = CountVectorizer(max_features=5000, lowercase=True, ngram_range=(1,2),analyzer = stemmed_words)
    #no lowercase
    # count = CountVectorizer(max_features=5000, ngram_range=(1,2),analyzer = stemmed_words)
    temp = count.fit_transform(x_train)
    tfidf = TfidfTransformer(use_idf=True, smooth_idf=True)
    temp2 = tfidf.fit_transform(temp)


    ### Support Vector Machine
    clf = eachModel['model']
    model= clf.fit(temp2,y_train)

    startTimePredict = time.time()
    prediction = model.predict(tfidf.transform(count.transform(x_test)))

    print("time taken for prediction: " + str((time.time() - startTimePredict)) + " secs")

    print("Model accuracy " + eachModel['model_name']+ ": " + str(np.mean(prediction==y_test)))

    #print("*"*10+ "Saving final model" + "*"*10 )
    # pickle.dump(model, open(filename, 'wb'))
    # pickle.dump(tfidf, open('model_sentiment/tfidf_trans.pk', 'wb'))
    # pickle.dump(count, open('model_sentiment/count_vert.pk', 'wb'))


    print('\nClasification report:\n', classification_report(y_test, prediction))
    print('\nConfussion matrix:\n',confusion_matrix(y_test, prediction)  )
        
    print("[End of Model] time taken: " + str((time.time() - startModelTime)) + " secs")
print("congrats thanks for your patience!!!")
print("time taken: " + str((time.time() - startTime)) + " secs")

# z_test = [input("What is your review? ")]
# prediction = model.predict(tfidf.transform(count.transform(z_test)))
# print("prediction is:" + str(prediction[0]))