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

"""# 2. Understanding the Corpus

So now that we've loaded our data and separated the word sense tag (*class label*) from the actual context (*document*), let's look at some statisctics. **Exercise: But as a quick exercise, kindly extract the following information**:

*   Number of documents in the dataset
*   Number of living_sense labels
*   Number of factory_sense labels
*   Also calculate the distribution of each class
*   Lastly, get the total number of words (no punctuations) for each class (*this one is a little hard*)
"""

# Write your code here! Feel free to search up NumPy tutorials

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

X_train, X_test, Y_train, Y_test = train_test_split(
    text, # Features
    sentiments, # Labels
    test_size = 0.3, # The defined test size; Training size is just 1 minus the test size
    random_state = 17 # So we can shuffle the elements, but with some consistency
)

print("=====\nTraining Data")
print("Document count: %s" % len(Y_train))
labels, counts = np.unique(Y_train, return_counts=True)
for label, count in zip(labels, counts):
  print("%s: %s (%.4f)" % (label, count, count/len(Y_train)))
        
print("=====\nTesting Data")
print("Document count: %s" % len(Y_test))
labels, counts = np.unique(Y_test, return_counts=True)
for label, count in zip(labels, counts):
  print("%s: %s (%.4f)" % (label, count, count/len(Y_test)))

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

x = vectorizer.fit_transform(X_train)
count_vect_df = pd.DataFrame(x.todense(), columns=vectorizer.get_feature_names())

print(count_vect_df.head())