#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 08:02:51 2017

@author: sam_hashem
"""

import json as jn
import sklearn 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np
import cls_flu_tweets as cft
import re
import nltk 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
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




###Preprocess tweets
def processTweet2(tweet):
    # process the tweets

    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet

def getStopWordList(stopWordListFileName):
    #read the stopwords file and build a list
    stopWords = stopwords.words('english')
    stopWords.append('AT_USER')
    stopWords.append('URL')



def getFeatureVector(tweet):
    featureVector = []
    #split tweet into words
    words = tweet.split()
    for w in words:
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        #ignore if it is a stop word
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector


## Loading the data
raw_data = cft.flu_tweets()

raw_data.load('tweets_training_data.txt','labels_training_data.txt')



data = {'screen_name': [],'created_at': [],'text': [],
        'retweet_count': [], 'favorite_count': [],
        'friends_count': [], 'followers_count': [],'labels':[]}

for tweet,labels in zip(raw_data.tweets,raw_data.labels):
    t = jn.loads(tweet)
    data['text'].append(t['text'])
    data['screen_name'].append(t['user']['screen_name'])
    data['created_at'].append(t['created_at'])
    data['retweet_count'].append(t['retweet_count'])
    data['favorite_count'].append(t['favorite_count'])
    data['friends_count'].append(t['user']['friends_count'])
    data['followers_count'].append(t['user']['followers_count'])
    data['labels'].append(labels)
    
    
df = pd.DataFrame(data)



tweets = []
featureList = []
for i in range(len(df)):
    sentiment = df['labels'][i]
    tweet = df['text'][i]
    processedTweet = processTweet2(tweet)
    featureVector = getFeatureVector(processedTweet)
    featureList.extend(featureVector)
    tweets.append((featureVector, sentiment))



def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features


featureList = list(set(featureList))
## End of preprocessing





##Naive Bayes Classifier

training_set = nltk.classify.util.apply_features\
(extract_features, tweets)
# Train the classifier Naive Bayes Classifier
NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
#ua is a dataframe containing all the united airline tweets
df['sentiement'] = \
df['tweet'].apply(lambda tweet: NBClassifier.classify\
  (extract_features(getFeatureVector(processTweet2(tweet)))))





