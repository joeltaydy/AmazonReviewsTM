from sklearn.feature_extraction.text import CountVectorizer
import nltk

def stemmed_words(doc):
    stemmer = nltk.stem.porter.PorterStemmer()

    analyzer = CountVectorizer().build_analyzer()

    return (stemmer.stem(w) for w in analyzer(doc))