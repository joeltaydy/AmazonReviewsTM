from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import nltk,re

def stemmed_words(doc):
    stemmer = nltk.stem.porter.PorterStemmer()

    analyzer = CountVectorizer().build_analyzer()
    return (stemmer.stem(w) for w in analyzer(doc))

def get_top_n_words(bow, vectorizer, n=10):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.
    
    get_top_n_words(["I love Python", "Python is a language programming", "Hello world", "I love the world"]) -> 
    [('python', 2),
     ('world', 2),
     ('love', 2),
     ('hello', 1),
     ('is', 1),
     ('programming', 1),
     ('the', 1),
     ('language', 1)]
    """
    #vec = CountVectorizer().fit(corpus)
    #bag_of_words = vec.transform(corpus)
    sum_words = bow.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    for word, freq in words_freq[:n]:
        print(word, freq) 
    return

def removeStopwords(content):
    
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words += ['phone','laptop','mobile','camera','the']
    toReturn = []
    for sent in content:
        stopped_review = ""
        for word in sent.split(" "):

            if word.lower() not in stop_words:
                stopped_review +=  word.lower()+ " "
        if stopped_review != "":
            toReturn.append(stopped_review)
    return toReturn
    