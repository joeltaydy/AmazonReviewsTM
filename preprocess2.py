import nltk, re, csv,time
from gensim import corpora
from random import shuffle
stop_list = nltk.corpus.stopwords.words('english')
# The following list is to further remove some frequent words in SGNews.
stop_list += ['would', 'said', 'say', 'year', 'day', 'also', 'first', 'last', 'one', 'two', 'people', 'told', 'new', 'could', 'singapore', 'three', 'may', 'like', 'world', 'since']
stemmer = nltk.stem.porter.PorterStemmer()


'''
parameters
file - File name  
field - field for the corpus. Could be review or product id
labelField - field for label (category or polarity)
lower - boolean
stop - boolean
stem - boolean
'''
def load_corpus(file, field,labelField, lower=True, stop=True, stem=True, sampleSize = None):
    # dir is a directory with plain text files to load.
    labels = []
    corpus = []
    
    '''
    Reading the whole set of data from file
    '''
    if sampleSize == None:
            
        with open(file,encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            counter =1
            for row in reader:
                labels.append(row[labelField])
                sent = nltk.word_tokenize(row[field])
                
                if lower:
                    sent = [w.lower() for w in sent]
                sent = [w for w in sent if re.search('^[a-z]+$', w)]

                if stop:
                    sent = [w for w in sent if w not in stop_list]
        
                if stem:
                    sent = [stemmer.stem(w) for w in sent]
        
                corpus.append(sent)
                if counter % 10000 ==0:
                    print(counter,sent)
                counter= counter+1

    else: #Extract n number of data from the excel
        with open(file,encoding='utf-8') as csvfile:
            sampleData = []
            reader = csv.DictReader(csvfile)
            counter =1
            for row in reader:
                sampleData.append(row)
            shuffle(sampleData) # Shuffle the data around
            sampleData=sampleData[:sampleSize]
            for row in sampleData:
                labels.append(row[labelField])
                sent = nltk.word_tokenize(row[field])
                
                if lower:
                    sent = [w.lower() for w in sent]
                sent = [w for w in sent if re.search('^[a-z]+$', w)]

                if stop:
                    sent = [w for w in sent if w not in stop_list]
        
                if stem:
                    sent = [stemmer.stem(w) for w in sent]
        
                corpus.append(sent)

    return labels, corpus

def docs2vecs(docs, dictionary):
    # docs is a list of documents returned by corpus2docs.
    # dictionary is a gensim.corpora.Dictionary object.
    vecs1 = [dictionary.doc2bow(doc) for doc in docs]
    return vecs1


startTime = time.time()
label, docs= load_corpus("data/preprocessed_reviewinfo.csv", field='Content',labelField='category',lower=True,stop=True,stem=True,sampleSize=500)
print('Finished reading sentences from the training data file. Time: ', time.time()-startTime )
print(len(docs))
print(docs[5])
all_tf_vectors = docs2vecs(docs, corpora.Dictionary(docs))
#print(all_tf_vectors)
print("finish converting to vector")
print(time.time() - startTime)