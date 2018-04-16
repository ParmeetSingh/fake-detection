import pandas as pd;
from nltk.stem import PorterStemmer
import re

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from textblob import TextBlob

ps = PorterStemmer()

#Rumour dataset
dataset = pd.read_csv("D:/Masters/Term 2/NLP/PROJECT/code/FinalDataset/rumour.csv",sep=",");

#regularExpression list
re_hashtags = '#[\w+]+'
re_usermentions = '@[\w+]+'
re_exlamationmarks = '!'
re_qsmark = '\?'
re_period = '\.'
re_urls = "(?P<url>https?://[^\s]+)"
#re_url = "(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
#remove stopwords
cachedStopWords = set(stopwords.words("english"))

def preprocessData(document):
     processedCorpus =[]    
     for line in document:
         line = line.lower()                        
         #remove punctuations
         line = re.sub('[^a-z]',' ', line)
         #split line
         words = line.strip().split()
         #stemming + removing stop words from a sentence
         filtered_sentence = ' '.join([ps.stem(w) for w in words if not w in cachedStopWords])
         #append to processedCorpus
         processedCorpus.append(filtered_sentence)
     return processedCorpus

def stanceDetection(tweets):
    stanceDetails = []
    for tweet in tweets:
    #print(tweet.text)
        analysis = TextBlob(tweet)
        #print(analysis.sentiment)
        stance = "neg"
        if(analysis.sentiment.polarity > 0):
            #positive sentence
            stance = "pos"
        elif(analysis.sentiment.polarity == 0):
            #neutral sentence
            stance = "neu"
        stanceDetails.append(stance)
    return stanceDetails

def find_ngrams(input_list, n):
    return list(zip(*[input_list[i:] for i in range(n)]))


import numpy as np
 
dataset["hashtags"] = dataset['text'].apply(lambda x: ' '.join(re.findall(re_hashtags, x)))
dataset["usermentions"] = 	dataset['text'].apply(lambda x: ' '.join(re.findall(re_usermentions, x)))

#removehashtags
dataset['processedText'] = dataset['text'].replace(re_hashtags, '', regex = True)
#removeusermentions
dataset['processedText'] = dataset['processedText'].replace(re_usermentions, '', regex = True)

dataset["capitalsCount"]  = dataset['processedText'].apply(lambda x: np.sum([1 for word in x.split() if word.isupper()]))
dataset["exclamationmarks"] = dataset['processedText'].apply(lambda x: len(re.findall(re_exlamationmarks, x)))
dataset["qsmarks"] = dataset['processedText'].apply(lambda x: len(re.findall(re_qsmark, x)))
dataset["periodcount"] = dataset['processedText'].apply(lambda x: len(re.findall(re_period, x)))
dataset["urls"] = dataset['processedText'].apply(lambda x: ' '.join(re.findall(re_urls, x)))


dataset['processedText'] = dataset['processedText'].replace(re_urls, '', regex = True)

dataset["processedText"] = preprocessData(dataset["processedText"])

dataset["sentiment_analysis"] = stanceDetection(dataset.text)


from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
vectorizercounts = count_vect.fit_transform(dataset["processedText"])

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
encodedData = dataset.apply(le.fit_transform)
dependantVar = encodedData.rumour.values
trainingdataset = encodedData[["urls","periodcount","exclamationmarks","qsmarks", "capitalsCount", "sentiment_analysis","hashtags","usermentions", "retweet_count","processedText","created","followers_count","verified","statuses_count","user_age"]]


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(trainingdataset, dependantVar, test_size = 0.33, random_state = 6)
X_train_countvector, X_test_countvector, y_train, y_test = train_test_split(vectorizercounts, dependantVar, test_size = 0.33, random_state = 6)


import scipy.sparse as sp
combined_train = sp.hstack([X_train,X_train_countvector])
combined_test = sp.hstack([X_test,X_test_countvector])


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
textclf =clf.fit(combined_train, y_train)
y_pred = textclf.predict(combined_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

#Out[165]: 0.8234986945169713
#Out[166]: 0.8135770234986945
#Out[167]: 0.8402088772845953
#Out[168]: 0.829242819843342
#Out[169]: 0.8234986945169713
#Out[170]: 0.8198433420365535
arr_accuracies = [0.8234986945169713, 0.8135770234986945, 0.8402088772845953, 0.829242819843342, 0.8234986945169713]
accuracies = np.array(arr_accuracies)
std = accuracies.std()
mean = accuracies.mean()