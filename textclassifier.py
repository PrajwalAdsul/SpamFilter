#!/usr/bin/env python
# coding: utf-8

# In[4]:


import sys
import pandas as pd
import numpy as np
import nltk
import sklearn
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# In[5]:


#Loading the sms dataset
df = pd.read_csv('SMSSPamCollection.tsv.txt', sep='\t', names=["Type", "sms"])
print(df.info())
print(df.head())


# In[6]:


#Checking class distribution
#print(df.describe())
print(df.describe(include='all'))
print('--------------')
type = df['Type']
print(df['Type'].value_counts())


# In[7]:


#Preprocessing data

from sklearn.preprocessing import LabelEncoder
#Converting ham and spam values to 0 and 1 
encoder = LabelEncoder()
Y = encoder.fit_transform(type)
print(Y[:10])


# In[8]:


# Storing sms data in another dataframe

messages = df['sms']
print(messages[:10])


# In[9]:


#PREPROCESSING

# Need to replace email addresses, URLs, numbers, etc in the text 
# so it can have some meaning rather than being a separate instance
# of itself

#will do this regular expressions
# http://regexlib.com

#Replacing email ids with 'emailid'
processed = messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','emailid')

#Replacing URLs with 'webaddr'
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','webaddr')

#Replacing money symbols with 'moneysymbol' 
processed = processed.str.replace(r'£|\$|\₹', 'moneysymb')

#Replacing 10digit phone numbers 
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','phonenum')

#Replacing any numbers (digits)
processed = processed.str.replace(r'\d+(\.\d+)?', 'num')


# In[10]:


#Removing punctuation
processed = processed.str.replace(r'[^\w\d\s]', ' ')

#Removing leading and trailing whitespace in a line of sms
processed = processed.str.replace(r'^\s+|\s+?$', '')

#Replacing whitespace between terms with a single space
processed = processed.str.replace(r'\s+', ' ')


# In[11]:


#Change all words to lower case 
processed = processed.str.lower()
print(processed)


# In[12]:


#Removing stop words from the corpus data

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = stopwords.words('english')


processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))

# #tokenizing words, easier to remove stop words
# words = []
# cleaned = []  #cleaned after removing stop words 
# for m in processed:
#     k = word_tokenize(m)
#     #print(k)
#     for w in k:
#         words.append(w)
        
# for word in words:
#     if word not in stop_words:
#         cleaned.append(word)

# print(cleaned)


# In[13]:


#Stemming the words to further utilize this data to the fullest

ps = nltk.PorterStemmer()
processed = processed.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))
# stemmed = []

# for word in cleaned:
#     word = ps.stem(word)
#     stemmed.append(word)
    
print(processed)


# In[14]:


words = []
for m in processed:
     k = word_tokenize(m)
     #print(k)
     for w in k:
        words.append(w)
words = nltk.FreqDist(words) #to get the freq distribution of words
print(words) 

print('Number of words:' , len(words))  #length of words
print('Most frequent words:',words.most_common(10)) #most frequently occuring words (top 10)
      


# In[15]:


#How to do feature selection here
#How to determine which words will be useful for differentiating betweem Spam and Ham
#Can use 1000 most frequent words as features


feature_words = list(words.keys())[:1000]


# In[16]:


#This method will find which of the 1000 word features are contained in messages

def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in feature_words:
        features[word] = (word in words)
    return features


features = find_features(processed[0])
# print(features)
# for key, value in features.items():
#     if value == True:
#         print(key)

messages = list(zip(processed, Y))
#Each sms text message will have the 1000 features to itself which will 
#tell if those 1000 features occured in the message or not (true or false)


# In[17]:


# #Defining seed for reproducibility
seed = 1
np.random.seed = seed
np.random.shuffle(messages)
print(messages)


# In[18]:


#Calling find_features function for each SMS message
featuresets = [(find_features(text), label) for (text, label) in messages]   
# print(featuresets)

# print(featuresets)




# In[19]:


#Splitting featuresets into training and testing datasetwsw using sklearn

from sklearn import model_selection
training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state=seed)


# In[20]:


print('Training set:',len(training))
print('Testing set:', len(testing))


# In[21]:


#Using sklearn algorithms 
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC


model = SklearnClassifier(SVC(kernel = 'linear'))

#Training the model on the training data
model.train(training)

#Testing on the testing data
accuracy = nltk.classify.accuracy(model, testing)*100
print("SVC Accuracy: {}".format(accuracy))


# In[33]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

#Defining models to train
names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier","Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = list(zip(names, classifiers))
# print(models)

for name, model in models:
    nltk_model = SklearnClassifier(model)
    nltk_model.train(training)
    print(nltk_model)
    accuracy = nltk.classify.accuracy(nltk_model, testing)*100
#     print("{} Accuracy: {}".format(name, accuracy))


# Ensemble method is a machine learning technique that combines several base models in order to produce one optimal predictive model.

# In[34]:


#Not satisfied with the accuracy produced by the classifiers used, will be using voting classifiers to further decide on the best classifier
#Ensemble methods - Voting classifier

from sklearn.ensemble import VotingClassifier

names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier","Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = list(zip(names, classifiers))

nltk_ensemble = SklearnClassifier(VotingClassifier(estimators = models, voting='hard', n_jobs = -1))
nltk_ensemble.train(training)
accuracy = nltk.classify.accuracy(nltk_model, testing)*100
print("Voting Classifier's accuracy:", accuracy)


# In[36]:


#Making class label predictions for testing dataset

text_features, labels = list(zip(*testing))
prediction = nltk_ensemble.classify_many(text_features)


# In[38]:


#Printing a confusion matrix and a classification report

print(classification_report(labels, prediction))

pd.DataFrame(
    confusion_matrix(labels, prediction),
    index = [['actual', 'actual'], ['ham', 'spam']],
    columns = [['predicted', 'predicted'], ['ham', 'spam']])


# In[ ]:


#THE END ------------ 

