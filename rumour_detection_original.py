
# coding: utf-8

# In[1]:


import json
import os
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import keras
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *

import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
import gensim.models.keyedvectors as word2vec
from gensim.models.keyedvectors import KeyedVectors

glove2word2vec(glove_input_file="glove.6B.50d.txt", word2vec_output_file="gensim_glove_vectors.txt")
glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)


# In[2]:


def filter_sentence(text):
    #tokenizer = RegexpTokenizer(r'\w+')
    tokenizer = RegexpTokenizer(r'[A-z]+')
    stop_words = set(stopwords.words('english'))
    word_tokens = [string.lower() for string in tokenizer.tokenize(text)]
    word_tokens = [w for w in word_tokens if not w in stop_words]
    #word_tokens = [w for w in word_tokens if len(w)>2]
    return word_tokens


# In[3]:


folder_names = ["charliehebdo","ferguson","germanwings-crash","ottawashooting","sydneysiege"]


# In[4]:


non_rumour_json_obj = []
for name in folder_names:
    non_rumours_paths = "pheme-rnr-dataset/"+name+"/non-rumours/"
    non_rumours_path = [x for x in os.listdir(non_rumours_paths)]
    for subfolder in non_rumours_path:
        subfolder_path = non_rumours_paths + subfolder + "/source-tweet/"
        json_files = [pos_json for pos_json in os.listdir(subfolder_path) if pos_json.endswith('.json')]
        for json_file in json_files:
            json_file_path = subfolder_path + json_file
            #print(json_file_path)
            with open(json_file_path) as json_data:
                d = json.load(json_data)
                non_rumour_json_obj.append(d)
rumour_json_obj = []
for name in folder_names:
    rumours_paths = "pheme-rnr-dataset/"+name+"/rumours/"
    rumours_path = [x for x in os.listdir(rumours_paths)]
    for subfolder in rumours_path:
        subfolder_path = rumours_paths + subfolder + "/source-tweet/"
        json_files = [pos_json for pos_json in os.listdir(subfolder_path) if pos_json.endswith('.json')]
        for json_file in json_files:
            json_file_path = subfolder_path + json_file
            #print(json_file_path)
            with open(json_file_path) as json_data:
                d = json.load(json_data)
                rumour_json_obj.append(d)


# In[5]:


retweet_count = []
text = []
created = []
followers_count = []
verified = []
statuses_count = []
user_age = []
rumour = []

for elem in non_rumour_json_obj:
    retweet_count.append(elem['retweet_count'])
    text.append(elem['text'])
    created.append(elem['created_at'])
    followers_count.append(elem['user']['followers_count'])
    verified.append(elem['user']['verified'])
    statuses_count.append(elem['user']['statuses_count'])
    user_age.append(elem['user']['created_at'])
    rumour.append('non_rumour')
for elem in rumour_json_obj:
    retweet_count.append(elem['retweet_count'])
    text.append(elem['text'])
    created.append(elem['created_at'])
    followers_count.append(elem['user']['followers_count'])
    verified.append(elem['user']['verified'])
    statuses_count.append(elem['user']['statuses_count'])
    user_age.append(elem['user']['created_at'])
    rumour.append('rumour')


# In[6]:


df_temp = pd.DataFrame()
df = pd.DataFrame()

df_temp['retweet_count'] = retweet_count
df_temp['text'] = text
df_temp['created'] = created
df_temp['followers_count'] = followers_count
df_temp['verified'] = verified
df_temp['statuses_count'] = statuses_count
df_temp['user_age'] = user_age
df_temp['rumour'] = rumour


# In[7]:


df.count()
df_temp.to_csv('/home/dell/rumour.csv',sep=',')


# In[8]:


le = preprocessing.LabelEncoder()
df_temp['verified_bool'] = le.fit_transform(df_temp['verified'])
df_temp['rumour_bool'] = le.fit_transform(df_temp['rumour'])
actual_labels = df_temp['rumour_bool']


# In[9]:


df = df_temp[['retweet_count','followers_count','verified_bool','statuses_count']]


# In[10]:


clf = RandomForestClassifier(max_depth=2)
X_train, X_test, y_train, y_test = train_test_split(df, actual_labels, test_size=0.33, random_state=42)


# In[11]:


clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(accuracy_score(pred,y_test))


# In[12]:


####next attempt


# In[13]:


tokens = []
for t in text:
    token_line = filter_sentence(t)
    tokens.append(token_line)
word_features = []

not_found = 0
text_features = []
for token_line in tokens:
    line = []
    for token in token_line:
            try: 
                word_vector = glove_model.get_vector(token)
                line.append(word_vector)
            except:
                not_found = not_found + 1
    text_features.append(line)


# In[14]:


##take average of features
avg_text_features = []
for lines in text_features:
    avg = np.zeros(50)
    count = 0
    for line in lines:
        avg = avg + line
        count = count + 1
    avg_text_features.append(avg)


# In[15]:


word_avg_features = np.array(avg_text_features)
concat_features = np.concatenate((word_avg_features,df.as_matrix()),axis=1)


# In[16]:


clf = RandomForestClassifier(max_depth=2)
X_train, X_test, y_train, y_test = train_test_split(word_avg_features, actual_labels, test_size=0.33, random_state=42)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(accuracy_score(pred,y_test))


# In[17]:


clf = RandomForestClassifier(max_depth=2)
X_train, X_test, y_train, y_test = train_test_split(concat_features, actual_labels, test_size=0.33, random_state=42)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(accuracy_score(pred,y_test))


# In[18]:


####LSTM approach


# In[19]:


#x = Input(shape=(32,))
#y = Dense(16, activation='softmax')(x)
#model = Model(x, y)
lengths = []
for line in text_features:
    lengths.append(len(line))
print(max(lengths))


# In[20]:


len(text_features)


# In[21]:


lengths = []
training_list_embedded = np.zeros(shape=(len(text_features),50,15))
for i in range(len(text_features)):
    sentence = text_features[i]
    length = len(sentence)
    padded_sequence = []
    j = 1
        
    while(1):
            #print(j)
            if j>(15-length):
                break
            padded_sequence.append(np.zeros(50))
            j = j + 1
    sentence = padded_sequence + sentence 
    #print(length,15-length)
    #print(np.array(sentence).transpose().shape)
    training_list_embedded[i,:,:] = np.array(sentence).transpose()[:,:15]
    #print(str(i) + " " + str(len(sentence)))


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(training_list_embedded, actual_labels, test_size=0.33, random_state=42)


# In[23]:


# create the model
model = Sequential()
model.add(LSTM(10,input_shape=(50,15,)))
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(training_list_embedded, np.array(actual_labels), nb_epoch=30, batch_size=100)


# In[24]:


model.evaluate(X_test,y_test)


# In[25]:


df.head()


# In[26]:


df_temp

