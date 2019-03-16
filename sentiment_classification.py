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

"""Before anything else, this is **Pandas**, easy-to-use data structures and data analysis tools for the Python. Think of it as an abstraction over NumPy and instead of treating your datasets as arrays, we can think of them as dataframes.

So in the code above, we used our count vectorizer and passed our training data on it. Fit transform just means, fit the vectorizer based on the vocab of the training data, so we can use a simple transform on the testing data later.

From the output data frame, we can tell that there are 4155 features (since there are 4155 columns). Each column is a feature and each row is a document. There are only 5 rows in the output because we printed only the head of the DataFrame.

**Question: Looking at our data, what is an issue here?**

(some) Answer:

*   **Number of features** - We have a load of features! 4k plus with only 130 training instances! This may be generalizable because we're account for many possible features, but we're seeing a lot of information. What happens if we expand out training samples, will out feature size grow? Yup!
*   **Sparcity** - Because we have a lot of features, we're to expect a lot of zeros as well. And if there is only one entry for a specific feature, it might not be super useful.

Let's look at how different configurations can affect the dataset.
"""

print("Default:", CountVectorizer().fit_transform(X_train).shape)
print("Min df set to 1%:", CountVectorizer(min_df=.01).fit_transform(X_train).shape)
print("Min df set to 5%:", CountVectorizer(min_df=.05).fit_transform(X_train).shape)
print("Min df (1%), Max df (90%):", CountVectorizer(max_df=.9, min_df=.01).fit_transform(X_train).shape)
print("Using max features (50):", CountVectorizer(max_features=50).fit_transform(X_train).shape)
print("Using 1,2,3-grams:", CountVectorizer(ngram_range=(1,3)).fit_transform(X_train).shape)
print("Using 1,2,3-grams, Min df (1%), Max df (90%):", CountVectorizer(max_df=.9, min_df=.01, ngram_range=(1,3)).fit_transform(X_train).shape)
print("Using a defined vocabulary:", CountVectorizer(vocabulary=['meme', 'of', 'the', 'week']).fit_transform(X_train).shape)
print("Lowercase off:", CountVectorizer(lowercase=False).fit_transform(X_train).shape)

"""See how the configurations affect the number of features we can learn from? There can be many ways to create representations that our algorithms can learn from. We just need to be smart and justify why our configurations should be considered.

Just note that extracting features is key! Instead of just counts, you can consider:

*   Term Frequency (TF)
*   Term Frequency Inverse Document Frequency (TFIDF)
*   Binary counts
*   POS counts
*   Lexicon counts

Explore the different levels of information we can extract from text. Some might not be as useful, while others will be fit for the problem.

# 6. Learning based on the features / Training a model / Testing a model

Now that we're able to extract information, let's start with our models. We won't go into detail with the models, but we'll consider the following:

*   Naive Bayes (NB)
*   k-Nearest Neighbors (kNN)
*   Support Vector Machines (SVM)

All ML algorithms have their own strentghs (training time, algorithmic complexicty, etc), so its yet another factor to consider when dealing with classification. So let's formalize our feature set as follows
"""

X_train, X_test, Y_train, Y_test = train_test_split(
    text, # Our training set
    sentiments, # Our test set
    test_size = 0.3, # The defined test size; Training size is just 1 minus the test size
    random_state = 12 # So we can shuffle the elements, but with some consistency
)

# We'll just use simple counts
vectorizer = CountVectorizer(
    max_df=.9, 
    min_df=.01,
    ngram_range=(1,1),
    binary=True
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

"""Let's look at the following lines of code:

*   `vectorizer.fit_transform(X_train)` - This function fits and transforms the data. Fit means to understand the data passed in. We passed `X_train`, so its like a "oh this is what the training data looks like, let's get its vocab!" Transform means, whatever the data was that was passed in, transform it based on the fit. This returns a fitted count matrix. By context, that also means you can peform a `vectorizer.fit(X_train)` which just fits the data.
*   `vectorizer.transform(X_test)` - This function just transforms based on the fit and since our vectorizer was already fitted with the `X_train`, we now just fit the `X_test` to the `X_train` vocab.

Now let's get to actual classification!
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# The three classifiers that we'll use
clfs = [
    KNeighborsClassifier(n_neighbors=5), 
    MultinomialNB(),
    svm.SVC(kernel='linear')
]

for clf in clfs:
  clf.fit(X_train, Y_train)
  y_pred = clf.predict(X_test)
  acc = accuracy_score(Y_test, y_pred)
  f1 = f1_score(Y_test, y_pred, average=None)
  
  print("%s\nAccuracy: %s\nF1 Score: %s\n=====" % (clf, acc, f1))