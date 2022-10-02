import nltk
import numpy as np
import re
import string
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer


nltk.download('twitter_samples')
nltk.download('stopwords')

positive_tweets_training_set = twitter_samples.strings('positive_tweets.json')
negative_tweets_training_set = twitter_samples.strings('negative_tweets.json')
neutral_tweets_training_set = []
sentiments = ['negative', 'positive', 'neutral']
sentiment_frequencies = {}

def preprocess(tweet):
    tweet = re.sub(r'[^a-zA-Z0-9]', '', tweet)
    tweet = re.sub(r'[0-9]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub('(@[A-Za-z0-9_]+)', '', tweet)
    return tweet
    

def tokenize(tweet):
    processed_tweet = []
    stemmer = PorterStemmer() 
    words_to_remove = stopwords.words('english')
    tweet_tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    tokens = tweet_tokenizer.tokenize(tweet)

      # remove stopwords and punctuation
    for token in tokens:
        if (token not in words_to_remove and token not in string.punctuation):
            final_word = stemmer.stem(token)   
            processed_tweet.append(final_word)
    return processed_tweet

def update_sentiment_count(tweet, sentiment):
    for word in tweet:
        if word in sentiment_frequencies:
            sentiment_frequencies[word][sentiment]+=1
        else:
            sentiment_frequencies[word] = [0, 0, 0]
            sentiment_frequencies[word][sentiment]+=1



