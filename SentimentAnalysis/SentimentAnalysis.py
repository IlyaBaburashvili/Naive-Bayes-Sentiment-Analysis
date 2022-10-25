from binascii import a2b_base64
import json
import nltk
import numpy as np
import pandas as pd
import re
import string
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import math



nltk.download('twitter_samples')
nltk.download('stopwords')

#positive_tweets_training_set = twitter_samples.strings('positive_tweets.json')
#negative_tweets_training_set = twitter_samples.strings('negative_tweets.json')
positive_tweets_training_set = []
negative_tweets_training_set = []
neutral_tweets_training_set = []
full_training_set = {}
positive_tweets_test_set = []
negative_tweets_test_set = []
neutral_tweets_test_set = []
full_test_set = {}
#sentiments = ['negative', 'positive', 'neutral']
sentiment_frequencies = {}
#total_words = total_negative = total_unique_negative = total_positive = total_unique_positive = total_neutral = total_unique_neutral = 0
index = 0
def read_train_csv(csv_filename):
    df = pd.read_csv(csv_filename, encoding='latin-1')
    tweets = df['textOriginal']
    sentiments = df['Sentiment']
    size = len(tweets)
    sentiment = -1
    for i in range(110000):
        #if np.isnan(sentiments[i]):
           # break
        sentiment = int(sentiments[i])
        if sentiment == 0:
            sentiment = 2
        elif sentiment == -1:
            sentiment = 0
        print(tweets[i])
        #if sentiments[i] == 'positive' or sentiments[i] == 1:
        #    sentiment = 1
        #elif sentiments[i] == 'negative' or sentiments[i] == 2:
        #    sentiment = 0
        #elif sentiments[i] == 'neutral' or sentiments[i] == 0:
         #   sentiment = 2
        tweet = preprocess(tweets[i])
        tweet_tokenized = tokenize(tweet)
        update_sentiment_count(tweet_tokenized, sentiment)
        if sentiment == 0:
            negative_tweets_training_set.append(tweet_tokenized)
            full_training_set[tweet_tokenized] = 0
        elif sentiment == 1:
            positive_tweets_training_set.append(tweet_tokenized)
            full_training_set[tweet_tokenized] = 1
        else:
            neutral_tweets_training_set.append(tweet_tokenized)
            full_training_set[tweet_tokenized] = 2


def read_test_csv(csv_filename):
    df = pd.read_csv(csv_filename, encoding='latin-1')
    tweets = df['textOriginal']
    sentiments = df['Sentiment']
    size = len(tweets)
    sentiment = -1
    for i in range(1, 1000):
        sentiment = int(sentiments[i])
        #if sentiment == 0:
        #    sentiment = 2
        #elif sentiment == -1:
        #    sentiment = 0
        #print(tweets[i])
        #if sentiments[i] == 'positive':
            #sentiment = 1
        #elif sentiments[i] == 'negative':
            #sentiment = 0
        #else:
            #sentiment = 2
        tweet = preprocess(tweets[i])
        tweet_tokenized = tokenize(tweet)
        if sentiment == 0:
            negative_tweets_test_set.append(tweet_tokenized)
            full_test_set[tweet_tokenized] = 0
        elif sentiment == 1:
            positive_tweets_test_set.append(tweet_tokenized)
            full_test_set[tweet_tokenized] = 1
        else:
            neutral_tweets_test_set.append(tweet_tokenized)
            full_test_set[tweet_tokenized] = 2

def read_json(json_filename):
    with open(json_filename) as json_file:
        all_word_probabilities = json.load(json_file)
        prior_probability_positive = all_word_probabilities['neg_pos_neutral_prob'][0]
        prior_probability_negative = all_word_probabilities['neg_pos_neutral_prob'][1]
        prior_probability_neutral = all_word_probabilities['neg_pos_neutral_prob'][2]
    return prior_probability_positive, prior_probability_negative, prior_probability_neutral, all_word_probabilities

def preprocess(tweet):
    print("first", tweet)
    tweet = re.sub(r'[^\x00-\x7F]', '', tweet) 
    tweet = re.sub(r'[^A-Za-z0-9 ]+', '', tweet)
    tweet = re.sub(r'[0-9]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub('(@[A-Za-z0-9_]+)', '', tweet)
    #print("fixed", tweet)
    return tweet
  

    
def tokenize(tweet):
    processed_tweet = []
    stemmer = PorterStemmer() 
    lemmatizer = WordNetLemmatizer()
    words_to_remove = stopwords.words('english')
    tweet_tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    tokens = tweet_tokenizer.tokenize(tweet)

      # remove stopwords and punctuation
    for token in tokens:
        if (token not in words_to_remove and token not in string.punctuation):
            token = token.lower()
            final_word = lemmatizer.lemmatize(token, pos="v")
            if final_word!=token:
                print(final_word, token)
            processed_tweet.append(final_word)
    print(processed_tweet)
    return tuple(processed_tweet)

def update_sentiment_count(tweet, sentiment):
    print(sentiment)
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

#def create_full_training_set(positive_training_set, negative_training_set, neutral_training_set):
#    for tweet in negative_training_set:
#        full_training_set[tweet] = 0
#    for tweet in positive_training_set:
#        full_training_set[tweet] = 1
#   for tweet in neutral_training_set:
#       full_training_set[tweet] = 2

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


def train_model():
    all_word_probabilities = {}
    total = len(full_training_set)
    prior_probability_positive = len(positive_tweets_training_set)/total
    prior_probability_negative = len(negative_tweets_training_set)/total
    prior_probability_neutral = len(neutral_tweets_training_set)/total
    total_unique_words, total_words_in_negative, total_words_in_positive, total_words_in_neutral = get_totals(sentiment_frequencies)
    for word in sentiment_frequencies:
        prob_word_negative = sentiment_frequencies[word][0]/total_words_in_negative
        prob_word_positive = sentiment_frequencies[word][1]/total_words_in_positive
        prob_word_neutral = sentiment_frequencies[word][2]/total_words_in_neutral
        all_word_probabilities[word] = [prob_word_negative, prob_word_positive, prob_word_neutral]
    all_word_probabilities['neg_pos_neutral_prob'] = [prior_probability_negative, prior_probability_positive, prior_probability_neutral]
    all_probablities_json = json.dumps(all_word_probabilities, indent=4)
    with open("probabilies3.json", "w") as outfile:
        outfile.write(all_probablities_json)
    return prior_probability_positive, prior_probability_negative, prior_probability_neutral, all_word_probabilities

def analyze_tweet(test_tweet, prior_probability_positive, prior_probability_negative, prior_probability_neutral, all_word_probabilities):
    #test_tweet = preprocess(test_tweet)
    #tokenized_tweet = tokenize(test_tweet)
    prob_tweet_positive=prior_probability_positive
    prob_tweet_negative=prior_probability_negative
    prob_tweet_neutral=prior_probability_neutral
    for word in test_tweet:
        if word in all_word_probabilities:
            prob_tweet_negative*=all_word_probabilities[word][0]
            prob_tweet_positive*=all_word_probabilities[word][1]
            prob_tweet_neutral*=all_word_probabilities[word][2]
        else:
            continue
            #prob_tweet_negative*=0.33
            #prob_tweet_positive*=0.33
            #prob_tweet_neutral*=0.33
    res = max(prob_tweet_negative, prob_tweet_positive, prob_tweet_neutral)
    if res == prob_tweet_negative:
        return 0
    elif res == prob_tweet_positive:
        return 1
    else:
        return 2


def test_model_accuracy(prior_probability_positive, prior_probability_negative, prior_probability_neutral, all_word_probabilities):
    total = len(full_test_set)
    correct = 0
    for tweet in full_test_set:
        res = analyze_tweet(tweet, prior_probability_positive, prior_probability_negative, prior_probability_neutral, all_word_probabilities)
        if res == full_test_set[tweet]:
            correct+=1
            print(tweet)
            print("Correct\n")
        else:
            print(tweet)
            print (res, full_test_set[tweet])
            print("Incorrect\n")
    return correct/total


def main():
    #read_train_csv('Tweets1.csv')
    #read_train_csv('Twitter_Data1.csv')
    a, b, c, d = read_json('probabilies3.json')
    read_test_csv('comments_Aiqa9l1vFNI.csv')
    #a, b, c, d = train_model()
    accuracy = test_model_accuracy(a, b, c, d)
    print(accuracy)

if __name__ == "__main__":
    main()
