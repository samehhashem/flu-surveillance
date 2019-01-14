#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 08:28:47 2017

@author: sam_hashem
"""

import pandas as pd
import numpy as np
import nltk
import string
from nltk.classify import NaiveBayesClassifier


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

is_positive = df['labels'].str.contains("pos")
is_negative = df['labels'].str.contains("neg")

positive_tweets = df[is_positive]
negative_tweets = df[is_negative]


## Set Useless Words


useless_words = nltk.corpus.stopwords.words("english") + list(string.punctuation)


def build_bag_of_words_features_filtered(words):
    return {
        word:1 for word in words \
        if not word in useless_words}



tokenized_negative_tweets = []
for text in negative_tweets['text']:
        tokenized_negative_tweets.append(nltk.word_tokenize(text))




negative_features = [
    (build_bag_of_words_features_filtered(text), 'neg') \
    for text in tokenized_negative_tweets
]


tokenized_positive_tweets = []
for text in positive_tweets['text']:
        tokenized_positive_tweets.append(nltk.word_tokenize(text))
        
        
   
positive_features = [
    (build_bag_of_words_features_filtered(text), 'pos') \
    for text in tokenized_positive_tweets
]     
        


split = 100

sentiment_classifier = NaiveBayesClassifier.train\
(positive_features[:split]+negative_features[:split])

positive_features_verify = positive_features[split:]
negative_features_verify = negative_features[split:]


nltk.classify.util.accuracy(sentiment_classifier, \
                            positive_features_verify+negative_features_verify)\
                            *100



