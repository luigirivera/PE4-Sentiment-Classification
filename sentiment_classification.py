import nltk
import os
import csv
import pandas as pd


nltk.download('punkt')

dataset = pd.read_csv('dataset/Virgin America and US Airways Tweets.csv', sep='\t') 

import numpy as np
dataset = np.array(dataset)

sentiments = dataset.T[0]
airline = dataset.T[1]
text = dataset.T[2]

print(sentiments[:5]) #Y in example
print(airline[:5])
print(text[:5]) #X in example

N = len(sentiments)
labels, counts = np.unique(sentiments, return_counts=True)
  
negative_indexes, dump = np.where(dataset == 'negative')
positive_indexes, dump = np.where(dataset == 'positive')
neutral_indexes, dump = np.where(dataset == 'neutral')

from nltk import word_tokenize
import string

def count_words(nArray):
  punct = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{}~'
  transtab = str.maketrans(dict.fromkeys(punct, ''))
  
  word_count = 0
  for sentence in nArray:
    tokens = word_tokenize(sentence.translate(transtab))
    # the translate portion just removed the punctuation
    word_count += len(tokens)
  return word_count

negative_count = count_words(text[negative_indexes])
positive_count = count_words(text[positive_indexes])
neutral_count = count_words(text[neutral_indexes])

print("Total document count: %s" % N)
for label, count, word_count in zip(labels, counts, [negative_count, positive_count, neutral_count]):
  print("%s: %s (%.4f); Word count: %s (%.4f)" % (label, count, count/N, word_count, word_count/np.sum([negative_count, positive_count, neutral_count])))

  punct = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{}~'
transtab = str.maketrans(dict.fromkeys(punct, ''))

print(text[6])

i = 0
while i < len(text):
  text[i] = text[i].translate(transtab)
  i += 1
  
print(text[6])

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

X_train_tfidf, X_test_tfidf, Y_train_tfidf, Y_test_tfidf = train_test_split(
    text, # Our training set
    sentiments, # Our test set
    test_size = 0.3, # The defined test size; Training size is just 1 minus the test size
    random_state = 12 # So we can shuffle the elements, but with some consistency
)
tfidf_vec = TfidfVectorizer(
  max_df=.9,
  min_df=.01,
  ngram_range=(1,1),
  binary=True
)
X_train_tfidf = tfidf_vec.fit_transform(X_train_tfidf)
X_test_tfidf = tfidf_vec.transform(X_test_tfidf)

X_train_cv, X_test_cv, Y_train_cv, Y_test_cv = train_test_split(
  text,
  sentiments,
  test_size = 0.3,
  random_state = 12
)
count_vec = CountVectorizer(
    max_df=.9, 
    min_df=.01,
    ngram_range=(1,1),
    binary=True
)
X_train_cv = count_vec.fit_transform(X_train_cv)
X_test_cv = count_vec.transform(X_test_cv)

X_train_tf, X_test_tf, Y_train_tf, Y_test_tf = train_test_split(
  text,
  sentiments,
  test_size = 0.3,
  random_state = 12
)
tf_vec = TfidfVectorizer(
  max_df=.9,
  min_df=.01,
  ngram_range=(1,1),
  binary=True,
  use_idf=False
)
X_train_tf = tf_vec.fit_transform(X_train_tf)
X_test_tf = tf_vec.transform(X_test_tf)

from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

mnb = MultinomialNB()
mnb.fit(X_train_tfidf, Y_train_tfidf)
y_pred = mnb.predict(X_test_tfidf)
acc = accuracy_score(Y_test_tfidf, y_pred)
f1 = f1_score(Y_test_tfidf, y_pred, labels="negative", average='macro')

print("\n\n%s\nTFIDF\nAccuracy: %s\nF1 Score: %s\n=====" % (mnb, acc, f1))

mnb = MultinomialNB()
mnb.fit(X_train_cv, Y_train_cv)
y_pred = mnb.predict(X_test_cv)
acc = accuracy_score(Y_test_cv, y_pred)
f1 = f1_score(Y_test_cv, y_pred, labels="negative", average='macro')

print("\n\n%s\nCount Vectorizer\nAccuracy: %s\nF1 Score: %s\n=====" % (mnb, acc, f1))

mnb = MultinomialNB()
mnb.fit(X_train_tf, Y_train_tf)
y_pred = mnb.predict(X_test_tf)
acc = accuracy_score(Y_test_tf, y_pred)
f1 = f1_score(Y_test_tf, y_pred, labels="negative", average='macro')

print("\n\n%s\nTerm Frequency\nAccuracy: %s\nF1 Score: %s\n=====" % (mnb, acc, f1))

