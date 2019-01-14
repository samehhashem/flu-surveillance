#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 17:53:05 2017

@author: sam_hashem
"""

import json as jn
import sklearn 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np
import cls_flu_tweets as cft
import operator
import re
import nltk 
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer('english')
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier







raw_data = cft.flu_tweets()

raw_data.load('tweets_training_data.txt','labels_training_data.txt')




data = {'screen_name': [],'created_at': [],'text': [],
        'retweet_count': [], 'favorite_count': [],
        'friends_count': [], 'followers_count': [],'labels':[]}

for tweet,label in zip(raw_data.tweets,raw_data.labels):
    t = jn.loads(tweet)
    data['text'].append(t['text'])
    data['screen_name'].append(t['user']['screen_name'])
    data['created_at'].append(t['created_at'])
    data['retweet_count'].append(t['retweet_count'])
    data['favorite_count'].append(t['favorite_count'])
    data['friends_count'].append(t['user']['friends_count'])
    data['followers_count'].append(t['user']['followers_count'])
    data['labels'].append(label)
    
    
df = pd.DataFrame(data)

def tweet_to_words(raw_tweet):
    letters_only = re.sub("[^a-zA-Z]", " ",raw_tweet) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] 
    return( " ".join( meaningful_words )) 
    
    
    
def clean_tweet_length(raw_tweet):
    letters_only = re.sub("[^a-zA-Z]", " ",raw_tweet) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] 
    return(len(meaningful_words))
    
df['label']=df['labels'].apply(lambda x: 0 if x=='neg' else 1)

df['clean_tweet']=df['text'].apply(lambda x: tweet_to_words(x))
df['Tweet_length']=df['text'].apply(lambda x: clean_tweet_length(x))
df['stem_tweet']=df['clean_tweet'].apply(lambda x: snowball_stemmer.stem(x))
df['lemma']=df['clean_tweet'].apply(lambda x: wordnet_lemmatizer.lemmatize(x))
df['tokenize'] = df['stem_tweet'].apply(lambda x: nltk.word_tokenize(x))




df_pos = pd.DataFrame()

df_pos = df[df['labels'] =='pos']

list_pos = []

for item in df_pos['tokenize']:
    list_pos.extend(item)

FreqDist_pos = nltk.FreqDist(list_pos)

pos_features = [word for (word, count) in FreqDist_pos.most_common(100)]

df_neg = pd.DataFrame()

df_neg = df[df['labels'] =='neg']

list_neg = []

for item in df_neg['tokenize']:
    list_neg.extend(item)

FreqDist_neg = nltk.FreqDist(list_neg)
neg_features = [word for (word, count) in FreqDist_neg.most_common(70)]


train,test = train_test_split(df,test_size=0.2,random_state=42)


all_feat = set(neg_features + pos_features)


dif_feat = list(set(pos_features) - set(neg_features))

train_clean_tweet=[]
for tweet in train['stem_tweet']:
    train_clean_tweet.append(tweet)
test_clean_tweet=[]
for tweet in test['stem_tweet']:
    test_clean_tweet.append(tweet)



v_dif = CountVectorizer()
v_dif.vocabulary = dif_feat
train_features_dif= v_dif.fit_transform(train_clean_tweet)
test_features_dif=v_dif.transform(test_clean_tweet)

v_all = CountVectorizer()
v_all.vocabulary = all_feat
train_features_all= v_all.fit_transform(train_clean_tweet)
test_features_all=v_all.transform(test_clean_tweet)


    
dense_features_dif=train_features_dif.toarray()
dense_test_dif= test_features_dif.toarray()
dense_features_all=train_features_all.toarray()
dense_test_all= test_features_all.toarray()
Accuracy=[]
Model=[]
Precision = []
Recall = []
clf1 = GaussianNB()
clf2 = MLPClassifier(hidden_layer_sizes=(20,20,20,20),max_iter=400)
clf3 = AdaBoostClassifier()

Classifiers = [
    GaussianNB(),
    MLPClassifier(hidden_layer_sizes=(20,20,20,20),max_iter=400),
    AdaBoostClassifier(),
    VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
    ]


for classifier in Classifiers:
        if classifier == GaussianNB():
            fit = classifier.fit(dense_features_dif,train['label'])
            pred_dif = fit.predict(dense_features_dif)
            accuracy = accuracy_score(pred_dif,test['label'])
            precision = precision_score(pred_dif,test['label'])
            recall = recall_score(pred_dif,test['label'])
            Accuracy.append(accuracy)
            Precision.append(precision)
            Model.append(classifier.__class__.__name__)
            report = classification_report(test['label'], pred_dif)
        else:
            fit = classifier.fit(dense_features_all,train['label'])
            pred_all = fit.predict(dense_features_all)
    
            accuracy = accuracy_score(pred_all,test['label'])
            precision = precision_score(pred_all,test['label'])
            recall = recall_score(pred_all,test['label'])
            Accuracy.append(accuracy)
            Precision.append(precision)
            Model.append(classifier.__class__.__name__)


            report = classification_report(test['label'], pred_all)
            print(report)
Index = [1,2,3,4]
plt.bar(Index,Accuracy)
plt.xticks(Index, Model,rotation=45)
plt.ylabel('output')
plt.xlabel('Model')
plt.title('Models')  

#
Index = [1,2,3,4]
plt.bar(Index,Precision)
plt.xticks(Index, Model,rotation=45)
plt.ylabel('output')
plt.xlabel('Model')