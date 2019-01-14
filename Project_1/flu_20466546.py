#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 20:58:12 2017

@author: sam_hashem
"""

import json as jn
import sklearn 
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import cls_flu_tweets as cft
import operator
import re
import nltk 
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer('english')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import pickle
from datetime import datetime
import pytz
import matplotlib.dates as mdates
import matplotlib.pyplot as plt



raw_data = cft.flu_tweets()


## Insert test file here
raw_data.load('tweets_training_data.txt')

## Loading to a dataframe
data = {'screen_name': [],'created_at': [],'text': [],
        'retweet_count': [], 'favorite_count': [],
        'friends_count': [], 'followers_count': []}

for tweet in raw_data.tweets:
    t = jn.loads(tweet)
    data['text'].append(t['text'])
    data['screen_name'].append(t['user']['screen_name'])
    data['created_at'].append(t['created_at'])
    data['retweet_count'].append(t['retweet_count'])
    data['favorite_count'].append(t['favorite_count'])
    data['friends_count'].append(t['user']['friends_count'])
    data['followers_count'].append(t['user']['followers_count'])


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
    
##Applying the functions onto the dataframe


df['clean_tweet']=df['text'].apply(lambda x: tweet_to_words(x))
df['Tweet_length']=df['text'].apply(lambda x: clean_tweet_length(x))
df['stem_tweet']=df['clean_tweet'].apply(lambda x: snowball_stemmer.stem(x))
df['tokenize'] = df['stem_tweet'].apply(lambda x: nltk.word_tokenize(x))
df['created_at'] = df['created_at'].apply(lambda x: pd.to_datetime(x))
df['Eastern'] = df['created_at'].apply(lambda x: pytz.timezone('Canada/Eastern').localize(x))
df['Eastern_date'] = df['Eastern'].apply(lambda x: datetime.date(x))

all_features_open = open('all_features_20466546.pkl', 'rb')

all_features = pickle.load(all_features_open)


classifier_open = open('flu_classifier_20466546.pkl','rb')
classifier = pickle.load(classifier_open)

test_clean_tweet=[]
for tweet in df['stem_tweet']:
    test_clean_tweet.append(tweet)

v = CountVectorizer()
v.vocabulary = all_features
X=v.transform(test_clean_tweet)
X_dense= X.toarray()

pred = classifier.predict(X_dense)


labellist = pred.tolist()

df['pred_label']= pred


pred_list =[]
for label in labellist:
    if label==1:
        pred_list.append('pos')
    else:
        pred_list.append('neg')
        
        
labelfile = open('labels_20466546.txt', 'w')
for item in pred_list:
  labelfile.write('%s\n' % item)
  
  
plt_df=df.groupby(['Eastern_date', 'pred_label']).size().unstack()

plt_df = plt_df.dropna()



plt_df = plt_df.reset_index()
plt_df['total'] = plt_df[0] + plt_df[1]
plt_df['percent'] = (plt_df[1] / plt_df['total']) * 100




plt.plot(plt_df.Eastern_date, plt_df.percent)

plt.title('Flu Surveilance system using twitter')
plt.xlabel('Date')
plt.ylabel('% positive flu tweets')
plt.legend(loc='best')
plt.show()


