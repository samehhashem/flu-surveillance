#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 19:58:22 2017

@author: sam_hashem

Script to solit the data into test and training data
"""

import cls_flu_tweets as cft


raw_data = cft.flu_tweets()


raw_data.load('tweets_training_data.txt','labels_training_data.txt')

def separate(cft.flu_tweets(), label):
    if label = 'neg':
        neg_data = cft.flu_tweets()
        for i in len(cft.flu_tweets().labels()):
            if cft.flu_tweets().labels[i] == 'neg':
                neg_data.tweets().append(cft.flu)
            
    