
# coding: utf-8

# In[2]:


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
from keras.layers import LSTM
from keras.models import *

import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense,Merge
from keras.layers import LSTM,RepeatVector
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
import gensim.models.keyedvectors as word2vec
from gensim.models.keyedvectors import KeyedVectors



# In[3]:


glove2word2vec(glove_input_file="glove.6B.50d.txt", word2vec_output_file="gensim_glove_vectors.txt")
glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)


# In[4]:


def filter_sentence(text):
    #tokenizer = RegexpTokenizer(r'\w+')
    tokenizer = RegexpTokenizer(r'[A-z]+')
    stop_words = set(stopwords.words('english'))
    word_tokens = [string.lower() for string in tokenizer.tokenize(text)]
    #word_tokens = [w for w in word_tokens if not w in stop_words]
    #word_tokens = [w for w in word_tokens if len(w)>2]
    return word_tokens


# In[5]:


def get_pd_frame_by_topics(topics):
    folder_names = topics
    

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


    df.count()
    df_temp.to_csv('/home/dell/rumour.csv',sep=',')

    temp = pd.to_datetime(df_temp.created, format='%a %b %d %H:%M:%S +0000 %Y')

    df_temp = df_temp.sort_values(by='created')

    le = preprocessing.LabelEncoder()
    df_temp['verified_bool'] = le.fit_transform(df_temp['verified'])
    df_temp['rumour_bool'] = le.fit_transform(df_temp['rumour'])
    actual_labels = df_temp['rumour_bool']

    df = df_temp[['retweet_count','followers_count','verified_bool','statuses_count']]
    return df,df_temp


# In[6]:


def convert_text_to_tokens(text,glove_model):
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

    # ##take average of features
    # avg_text_features = []
    # for lines in text_features:
    #     avg = np.zeros(50)
    #     count = 0
    #     for line in lines:
    #         avg = avg + line
    #         count = count + 1
    #     avg_text_features.append(avg)
    # word_avg_features = np.array(avg_text_features)
    # concat_features = np.concatenate((word_avg_features,df_text.as_matrix()),axis=1)

    lengths = []
    for line in text_features:
        lengths.append(len(line))
    print(max(lengths))


    lengths = []
    training_list_embedded = np.zeros(shape=(len(text_features),50,20))
    for i in range(len(text_features)):
        sentence = text_features[i]
        length = len(sentence)
        padded_sequence = []
        j = 1

        while(1):
                #print(j)
                if j>(20-length):
                    break
                padded_sequence.append(np.zeros(50))
                j = j + 1
        sentence = padded_sequence + sentence 
        training_list_embedded[i,:,:] = np.array(sentence).transpose()[:,:20]
    return training_list_embedded


# In[7]:


####LSTM approach
def stacked_lstm(X_train,y_train,bsize):
    length =int(len(X_train)/bsize)*bsize


    # create the model
    inp1 = Input(batch_shape=(bsize,50,20),name='input1')
    input1 = Permute((2,1))(inp1)
    m1 = LSTM(20,input_shape=(20,50),stateful=True,batch_size=bsize,name='model1',recurrent_dropout=0.5)(input1)
    
    inp2 = Input(batch_shape=(bsize,50,20),name='input2')
    input2 = Permute((2,1))(inp2)
    m2 = LSTM(20,input_shape=(20,50),stateful=True,batch_size=bsize,name='model2',recurrent_dropout=0.5)(input2)
    
    inp3 = Input(batch_shape=(bsize,50,20),name='input3')
    input3 = Permute((2,1))(inp3)
    m3 = LSTM(20,input_shape=(20,50),stateful=True,batch_size=bsize,name='model3',recurrent_dropout=0.5)(input3)
    
    inp4 = Input(batch_shape=(bsize,50,20),name='input4')
    input4 = Permute((2,1))(inp4)
    m4 = LSTM(20,input_shape=(20,50),stateful=True,batch_size=bsize,name='model4',recurrent_dropout=0.5)(input4)
    
    inp5 = Input(batch_shape=(bsize,50,20),name='input5')
    input5 = Permute((2,1))(inp5)
    m5 = LSTM(20,input_shape=(20,50),stateful=True,batch_size=bsize,name='model5',recurrent_dropout=0.5)(input5)
    
    inp = Input(batch_shape=(bsize,50,20),name='weight_model')
    input_c = Permute((2,1))(inp)
    w_model = LSTM(20,input_shape=(20,50),stateful=False,batch_size=bsize,recurrent_dropout=0.5)(input_c)
    w_model = Dense(5, activation='softmax')(w_model)
    w_model = Dropout(0.5)(w_model)
    merge_softmax = Reshape((1, 5))(w_model)
    
    merged = merge([m1, m2 ,m3 ,m4, m5],mode='concat')
    reshaped = Reshape((5, 20))(merged)
    
    final_merged = merge([reshaped, merge_softmax], mode = 'dot', dot_axes = [1, 2])
    reshaped2 = Reshape((20, ))(final_merged)

    
    reshaped2 = Dense(10)(reshaped2)
    reshaped2 = Dropout(0.5)(reshaped2)
    main_output = Dense(1, activation='sigmoid',name='main_output')(reshaped2)
    model = Model(inputs=[inp1,inp2,inp3,inp4,inp5,inp],outputs=[main_output])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    #model.fit([X_train[0:length]], np.array(y_train)[0:length], nb_epoch=40, batch_size=bsize)
    return model


# In[8]:


#folder_names = ["ferguson"]
folder_names = ["charliehebdo","ferguson","germanwings-crash","ottawashooting","sydneysiege"]
df_features,df_text = get_pd_frame_by_topics(folder_names)
all_labels = df_text['rumour_bool']
all_data = convert_text_to_tokens(df_text['text'],glove_model)
X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.33, random_state=42,shuffle = False)
bsize = 100
length_train =int(len(X_train)/bsize)*bsize
length_test =int(len(X_test)/bsize)*bsize
imp = X_train[0:length_train]
impt = X_test[0:length_test]
model = stacked_lstm(X_train,y_train,bsize)
#model.fit([imp,imp,imp,imp,imp,imp], np.array(y_train)[0:length], nb_epoch=30, batch_size=bsize)
history = model.fit([imp,imp,imp,imp,imp,imp], np.array(y_train)[0:length_train], nb_epoch=15, batch_size=bsize,validation_data=([impt,impt,impt,impt,impt,impt],y_test[0:length_test]))
import pickle
f = open('multilstm_context_stacked.pckl', 'wb')
pickle.dump(history.history, f)
f.close()


# In[9]:


#folder_names = ["ferguson"]
folder_names = ["charliehebdo","ferguson","germanwings-crash","ottawashooting","sydneysiege"]
df_features,df_text = get_pd_frame_by_topics(folder_names)
all_labels = df_text['rumour_bool']
all_data = convert_text_to_tokens(df_text['text'],glove_model)
X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.33, random_state=42,shuffle = False)
K = 5
accuracies = []
for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.33,shuffle = False)
    bsize = 100
    length_train =int(len(X_train)/bsize)*bsize
    length_test =int(len(X_test)/bsize)*bsize
    model = stacked_lstm(X_train,y_train,bsize)
    imp = X_train[0:length_train]
    model.fit([imp,imp,imp,imp,imp,imp], np.array(y_train)[0:length_train], nb_epoch=15, batch_size=bsize)
    inp = X_test[0:length_test]
    loss,acc1 = model.evaluate([inp,inp,inp,inp,inp,inp],y_test[0:length_test],batch_size=bsize)
    accuracies.append(acc1)


# In[11]:


print(np.mean(accuracies))
print(np.std(accuracies))
print(accuracies)


# In[ ]:


bsize = 100
length =int(len(X_test)/bsize)*bsize
inp = X_test[0:length]
model.evaluate([inp,inp,inp,inp,inp,inp],y_test[0:length],batch_size=bsize)


# In[ ]:


folder_names = ["charliehebdo"]
#folder_names = ["charliehebdo","ferguson","germanwings-crash","ottawashooting","sydneysiege"]
df_features,df_text = get_pd_frame_by_topics(folder_names)
germanwingscrash_labels = df_text['rumour_bool']
germanwingscrash_data = convert_text_to_tokens(df_text['text'],glove_model)
bsize = 100
length =int(len(germanwingscrash_labels)/bsize)*bsize
inp = germanwingscrash_data[0:length]
X_train, X_test, y_train, y_test = train_test_split(germanwingscrash_data,germanwingscrash_labels, test_size=0.33, random_state=42,shuffle = False)
model = stacked_lstm(X_train,y_train,bsize)
bsize = 100
length =int(len(X_test)/bsize)*bsize
inp = X_test[0:length]
model.evaluate([inp,inp,inp,inp,inp,inp],y_test[0:length],batch_size=bsize)


# In[ ]:


folder_names = ["ferguson"]
#folder_names = ["charliehebdo","ferguson","germanwings-crash","ottawashooting","sydneysiege"]
df_features,df_text = get_pd_frame_by_topics(folder_names)
germanwingscrash_labels = df_text['rumour_bool']
germanwingscrash_data = convert_text_to_tokens(df_text['text'],glove_model)
bsize = 100
length =int(len(germanwingscrash_labels)/bsize)*bsize
inp = germanwingscrash_data[0:length]
X_train, X_test, y_train, y_test = train_test_split(germanwingscrash_data,germanwingscrash_labels, test_size=0.33, random_state=42,shuffle = False)
model = stacked_lstm(X_train,y_train,bsize)
bsize = 100
length =int(len(X_test)/bsize)*bsize
inp = X_test[0:length]
model.evaluate([inp,inp,inp,inp,inp,inp],y_test[0:length],batch_size=bsize)


# In[144]:


# clf = RandomForestClassifier(max_depth=2)
# X_train, X_test, y_train, y_test = train_test_split(df, actual_labels, test_size=0.33, random_state=42)


# In[163]:


# clf.fit(X_train, y_train)
# pred = clf.predict(X_test)
# print(accuracy_score(pred,y_test))


# In[146]:


####next attempt


# In[42]:


df_text['text']

