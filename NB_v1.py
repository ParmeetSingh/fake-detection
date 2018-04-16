import pandas as pd;
import nltk;
from nltk.corpus import stopwords;
import string;
from string import punctuation;
from nltk.stem import PorterStemmer
import nltk
import re
from collections import Counter
import numpy as np



# Read Data
df_data = pd.read_csv("D:/Masters/Term 2/NLP/PROJECT/code/FinalDataset/rumour.csv",sep=",")
# Fill in empty values.
df_data = df_data.fillna('')

# one time initialization:: Don't Run this
nltk.download()
df_data.info;
df_data.head();
df_data.language.unique()

#Preprocessing the stop words, white spaces and tabs
stop = stopwords.words('english')
   
def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def removewhites(text):
    re.sub( '\s+', ' ', text ).strip()
    return text

def find_ngrams(input_list, n):
    return list(zip(*[input_list[i:] for i in range(n)]))

df_data['text'] = df_data['text'].replace('\d+', '', regex = True)
df_data['text'] = df_data['text'].replace('[^\w\s\+]', '', regex = True)


df_data.text.fillna("", inplace=True);
df_data['text']=df_data.text.apply(remove_punctuations)
df_data['text']=df_data.text.apply(removewhites)

#Stemming
ps = PorterStemmer()
df_data['text'] = df_data['text'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split() if ps.stem(word) not in (stop)]))

#Extracting labels
labels = df_data.iloc[:,8].values

#Preparing the training data.
df_tt= df_data[["retweet_count","text","created","followers_count","verified","statuses_count","user_age"]]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import scipy.sparse as sp
from scipy import sparse

#TFIDF , Count vectorizer and ngrams(unigrams , bigrams , trigrams) extracted
tfidf = TfidfVectorizer()
tfidfvector = tfidf.fit_transform(df_tt['text'],df_tt['retweet_count'])

#count vectorizer 
count_vect = CountVectorizer()
vectorizercounts = count_vect.fit_transform(df_tt['text'],df_tt['retweet_count'])

ngrams = CountVectorizer(ngram_range=(2, 3))
vectorizercountsngrams = ngrams.fit_transform(df_tt['text'],df_tt['retweet_count'])

# Splitting data with the same random state
X_train_countvector, X_test_countvector, y_train, y_test = train_test_split(vectorizercounts, labels, test_size = 0.33, random_state = 7)
X_train_tfidfvector, X_test_tfidfvector, y_train, y_test = train_test_split(tfidfvector, labels, test_size = 0.33, random_state = 7)
X_train_ngrams, X_test_ngrams, y_train, y_test = train_test_split(vectorizercountsngrams, labels, test_size = 0.33, random_state = 7)

#Combining the features into a stack
combined_train = sp.hstack([X_train_countvector, X_train_tfidfvector,X_train_ngrams])

#Classifier used 
classifier = MultinomialNB()
text_clf = classifier.fit(combined_train, y_train)

combined_test = sp.hstack([X_test_countvector, X_test_tfidfvector,X_test_ngrams])

predicted=text_clf.predict(combined_test)

# Accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(predicted, y_test)