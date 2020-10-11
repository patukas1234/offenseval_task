import numpy as np
import pandas as pd
import emoji
import nltk
import re

from collections import Counter, defaultdict
from wordsegment import load, segment
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.preprocessing import LabelEncoder

load()


def encode_train_labels(train_data, subtask):

    le = LabelEncoder()

    subtask_labels = train_data[subtask].dropna()

    encoded = le.fit_transform(subtask_labels)

    return encoded, train_data.tweet[subtask_labels.index].tolist()

def encode_test_labels(labels):

    le = LabelEncoder()

    encoded = le.fit_transform(labels)

    return encoded




def preproc_one_tweet(tweet, tokenizer):

    tweet = tweet.lower()
    tweet = emoji.demojize(tweet, use_aliases=True)
    
    tweet = ' '.join(list(map(lambda tok: ' '.join(segment(tok)) if re.search(r"[#_]+", tok) else tok,tweet.split())))
    tweet = re.sub(r"_", " ", tweet)
    tweet = tokenizer.tokenize(tweet)
    tweet = list(filter(lambda tok: re.search(r"[\w!']+", tok), tweet))
    
    tweet = ' '.join(tweet)
    
    #TODO: add a preproc clause based on model

    return tweet



def preproc_tweets(tweets, remove_handle):

    tokenizer = TweetTokenizer(reduce_len=True, strip_handles= remove_handle)

    cleaned_tweets = [preproc_one_tweet(tweet, tokenizer) for tweet in tweets]

    return cleaned_tweets


def token_count(tweets, top_n, language):

    stop_words = list(set(stopwords.words(language)))
    
    flattened_tweets = [tok for tweet in tweets for tok in tweet.split() if tok not in stop_words]
    
    token_freq = dict(Counter(flattened_tweets).most_common(top_n))
    return token_freq



class model():
    def __init__(self, classifier, task):
        self.classifier = classifier
        
        self.task = task
        self.subtask = "sub" + task

 



    
