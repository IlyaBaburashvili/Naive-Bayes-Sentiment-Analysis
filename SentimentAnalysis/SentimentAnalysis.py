import nltk
import numpy as np
import re
import string
import pandas as pd
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer


nltk.download('twitter_samples')
nltk.download('stopwords')

#positive_tweets_training_set = twitter_samples.strings('positive_tweets.json')
#negative_tweets_training_set = twitter_samples.strings('negative_tweets.json')
positive_tweets_training_set = []
negative_tweets_training_set = []
neutral_tweets_training_set = []
full_training_set = {}
sentiments = ['negative', 'positive', 'neutral']
sentiment_frequencies = {}
total_words = total_negative = total_unique_negative = total_positive = total_unique_positive = total_neutral = total_unique_neutral = 0

def read_csv(csv_filename):
    df = pd.read_csv(csv_filename)
    tweets = df['textOriginal']
    sentiments = df['Sentiment']
    for tweet in tweets:
        if sentiments[tweet] == 0:
            negative_tweets_training_set += tweets[tweet]
        elif sentiments[tweet] == 1:
            positive_tweets_training_set += tweets[tweet]
        else:
            neutral_tweets_training_set += tweets[tweet]

def preprocess(tweet):
    tweet = re.sub(r'[^\x00-\x7F]', '', tweet) 
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
            sentiment_frequencies[word] = [1, 1, 1]  #all start with 1 to avoid probability of 0
            sentiment_frequencies[word][sentiment]+=1

#def get_all_sentiments(positive_tweets, negative_tweets, neutral_tweets):
 #   all_sentiments = np.array([])
  #  all_sentiments = np.append(all_sentiments, np.ones(len(positive_tweets)))
  #  all_sentiments = np.append(all_sentiments, np.zeros(len(negative_tweets)))
  #  all_sentiments = np.append(all_sentiments, 2(len(neutral_tweets)))
  #  return all_sentiments

def create_full_training_set(positive_training_set, negative_training_set, neutral_training_set):
    for tweet in negative_training_set:
        full_training_set[tweet] = 0
    for tweet in positive_training_set:
        full_training_set[tweet] = 1
    for tweet in neutral_training_set:
        full_training_set[tweet] = 2

def get_totals(sentiment_frequencies):
    total_unique_words = len(sentiment_frequencies)
    #total_unique_negative = 0
    #total_unique_positive = 0
    #total_unique_neutral = 0
    total_words_in_positive = 0
    total_words_in_negative = 0
    total_words_in_neutral = 0
    for word in sentiment_frequencies:
        total_words_in_negative+=sentiment_frequencies[word][0]
        total_words_in_positive+=sentiment_frequencies[word][1]
        total_words_in_neutral+=sentiment_frequencies[word][2]
    return total_unique_words, total_words_in_negative, total_words_in_positive, total_words_in_neutral


def train_model(sentiment_frequencies, full_training_set_clean):
    all_word_probabilities = {}
    total = len(full_training_set_clean)
    probability_positive = len(positive_tweets_training_set)/total
    probability_negative = len(negative_tweets_training_set)/total
    probability_neutral = len(neutral_tweets_training_set)/total
    total_unique_words, total_words_in_negative, total_words_in_positive, total_words_in_neutral = get_totals(sentiment_frequencies)
    for word in sentiment_frequencies:
        prob_word_negative = sentiment_frequencies[word][0]/total_words_in_negative
        prob_word_positive = sentiment_frequencies[word][1]/total_words_in_positive
        prob_word_neutral = sentiment_frequencies[word][2]/total_words_in_neutral
        all_word_probabilities[word] = [prob_word_negative, prob_word_positive, prob_word_neutral]
    return probability_positive, probability_negative, probability_neutral, all_word_probabilities

def analyze_tweet(test_tweet, probability_positive, probability_negative, probability_neutral, all_word_probabilities):
    test_tweet = preprocess(test_tweet)
    tokenized_tweet = tokenize(test_tweet)
    for word in tokenized_tweet:
        if word in all_word_probabilities:
