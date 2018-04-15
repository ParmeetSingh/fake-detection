
# coding: utf-8

# In[1]:


import pickle
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:


f = open('attention.pckl', 'rb')
obj = pickle.load(f)
attention_testing = obj['val_acc']
attention_training = obj['acc']
f.close()


# In[3]:


f = open('conditioned.pckl', 'rb')
obj = pickle.load(f)
conditioned_testing = obj['val_acc']
conditioned_training = obj['acc']
f.close()


# In[4]:


f = open('stacked_lstm.pckl', 'rb')
obj = pickle.load(f)
stacked_reactions_testing = obj['val_acc']
stacked_reactions_training = obj['acc']
f.close()


# In[5]:


f = open('context_stacked.pckl', 'rb')
obj = pickle.load(f)
context_stacked_testing = obj['val_acc']
context_stacked_training = obj['acc']
f.close()


# In[11]:


plt.plot(attention_testing)
#plt.plot(conditioned_testing)
plt.plot(stacked_reactions_testing)
plt.plot(context_stacked_testing)


# In[7]:


#folder_names = ["charliehebdo","ferguson","germanwings-crash","ottawashooting","sydneysiege"]

f = open('multilstm_context_stacked.pckl', 'rb')
obj = pickle.load(f)
multilstm_context_stacked_testing = obj['val_acc']
multilstm_context_stacked_training = obj['acc']
f.close()

f = open('charlie_hebdo_context_stacked.pckl', 'rb')
obj = pickle.load(f)
charliehebdo_context_stacked_testing = obj['val_acc']
charliehebdo_context_stacked_training = obj['acc']
f.close()
f = open('ferguson_context_stacked.pckl', 'rb')
obj = pickle.load(f)
ferguson_context_stacked_testing = obj['val_acc']
ferguson_context_stacked_training = obj['acc']
f.close()
f = open('germanwings-crash_context_stacked.pckl', 'rb')
obj = pickle.load(f)
germanwings_context_stacked_testing = obj['val_acc']
germanwings_context_stacked_training = obj['acc']
f.close()
f = open('ottawashooting_context_stacked.pckl', 'rb')
obj = pickle.load(f)
ottawashooting_context_stacked_testing = obj['val_acc']
ottawashooting_context_stacked_training = obj['acc']
f.close()
f = open('sydneysiege_context_stacked.pckl', 'rb')
obj = pickle.load(f)
sydneysiege_context_stacked_testing = obj['val_acc']
sydneysiege_context_stacked_training = obj['acc']
f.close()


# In[8]:


plt.plot(multilstm_context_stacked_testing,label="multicontext")
plt.plot(context_stacked_testing,label="context")
# plt.plot(attention_testing,label='attention')
# plt.plot(conditioned_testing,label='conditioned')
# plt.plot(stacked_reactions_testing,label='stacked_reactions')
# plt.plot(context_stacked_testing,label='content_stacked')
plt.legend()


# In[13]:


attention_with_conditional_encoding_reactions = [0.7636842100243819, 0.7563157897246512, 0.7615789488742226, 0.7436842102753488, 0.7557894713000247]
attention_reactions = [0.772105264036279, 0.7910526332102323, 0.7494736878495467, 0.7857894709235743, 0.7710526334611993]
stacked_reactions = [0.7726315824609054, 0.7652631621611746, 0.7400000001254835, 0.7757894616377982, 0.7647368468736347]
conditional_encoding_reactions = [0.7573684234368173, 0.7689473754481265, 0.7752631557615179, 0.7605263088878832, 0.7589473692994368]
stacked_context = [0.7710526334611993, 0.7594736814498901, 0.7784210568980167, 0.7547368344507719, 0.7510526337121662]
multi_lstm = [0.7605263120249698, 0.7689473785852131, 0.75, 0.7578947387243572, 0.7584210508748105]


# In[17]:


from scipy import stats
print(stats.ttest_ind(attention_with_conditional_encoding_reactions,attention_reactions,equal_var=False))
print(stats.ttest_ind(stacked_reactions,attention_reactions,equal_var=False))
print(stats.ttest_ind(conditional_encoding_reactions,attention_reactions,equal_var=False))
print(stats.ttest_ind(stacked_context,attention_reactions,equal_var=False))
print(stats.ttest_ind(multi_lstm,attention_reactions,equal_var=False))

