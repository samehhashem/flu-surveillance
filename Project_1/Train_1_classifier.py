#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 20:01:30 2017

@author: sam_hashem
"""

import json as jn
import sklearn 
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.cross_validation
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
from sklearn.cross_validation import train_test_split
import pytz
from datetime import datetime




## Loading dat into class: flu_tweets()
raw_data = cft.flu_tweets()

raw_data.load('tweets_training_data.txt','labels_training_data.txt')



##Creating the colomnds for a dataframe
data = {'screen_name': [],'created_at': [],'text': [],
        'retweet_count': [], 'favorite_count': [],
        'friends_count': [], 'followers_count': [],'labels':[],'Eastern':[]}


##Loading things into the dataframe
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
    current_date = datetime.strptime(tweet[15:45], '%a %b %d %H:%M:%S +0000 %Y')
    current_date = pytz.timezone("Canada/Eastern").localize(current_date)
    current_date = datetime.date(current_date)
    data['Eastern'].append(current_date)
    
    
df = pd.DataFrame(data)

## Cleaning a tweet
def tweet_to_words(raw_tweet):
    letters_only = re.sub("[^a-zA-Z]", " ",raw_tweet) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] 
    return( " ".join( meaningful_words )) 
    
    
## Clean Tweet Length
def clean_tweet_length(raw_tweet):
    letters_only = re.sub("[^a-zA-Z]", " ",raw_tweet) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] 
    return(len(meaningful_words))
    
    
## Make labels as 1 and zero
    
df['label']=df['labels'].apply(lambda x: 0 if x=='neg' else 1)


##Applying the functions onto the dataframe


df['clean_tweet']=df['text'].apply(lambda x: tweet_to_words(x))
df['Tweet_length']=df['text'].apply(lambda x: clean_tweet_length(x))
df['stem_tweet']=df['clean_tweet'].apply(lambda x: snowball_stemmer.stem(x))
df['lemma']=df['clean_tweet'].apply(lambda x: wordnet_lemmatizer.lemmatize(x))
df['tokenize'] = df['stem_tweet'].apply(lambda x: nltk.word_tokenize(x))
df['created_at'] = df['created_at'].apply(lambda x: pd.to_datetime(x))
df['Eastern'] = df['created_at'].apply(lambda x: pytz.timezone('Canada/Eastern').localize(x))



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
neg_features = [word for (word, count) in FreqDist_neg.most_common(100)]



all_feat = set(neg_features + pos_features)
with open('all_features_20466546.pkl', 'wb') as f:
    pickle.dump(all_feat, f)
f.close()




train_clean_tweet=[]
for tweet in df['stem_tweet']:
    train_clean_tweet.append(tweet)
    
v = CountVectorizer()
v.vocabulary = all_feat
train_features= v.fit_transform(train_clean_tweet)


Model = MLPClassifier(hidden_layer_sizes=(20,20,20,20,20),max_iter=400)

Model.fit(train_features,df['label'])


import pickle
# now you can save it to a file
with open('flu_classifier_20466546.pkl', 'wb') as f:
    pickle.dump(Model, f)
f.close()
    