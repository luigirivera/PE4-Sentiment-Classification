import nltk
import os
import csv
import pandas as pd
import numpy as np

nltk.download('punkt')

dataset = pd.read_csv('dataset/Virgin America and US Airways Tweets.csv', sep='\t') 

dataset = np.array(dataset)

sentiments = dataset.T[0]
airline = dataset.T[1]
text = dataset.T[2]

print(sentiments[:5])
print(airline[:5])
print(text[:5])

"""# 2. Understanding the Corpus

So now that we've loaded our data and separated the word sense tag (*class label*) from the actual context (*document*), let's look at some statisctics. **Exercise: But as a quick exercise, kindly extract the following information**:

*   Number of documents in the dataset
*   Number of living_sense labels
*   Number of factory_sense labels
*   Also calculate the distribution of each class
*   Lastly, get the total number of words (no punctuations) for each class (*this one is a little hard*)
"""

# Write your code here! Feel free to search up NumPy tutorials

N = len(Y)
labels, counts = np.unique(Y, return_counts=True)
  
living_indexes, dump = np.where(dataset == 'living_sense')
factory_indexes, dump = np.where(dataset == 'factory_sense')

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

living_count = count_words(X[living_indexes])
factory_count = count_words(X[factory_indexes])

print("Total document count: %s" % N)
for label, count, word_count in zip(labels, counts, [factory_count, living_count]):
  print("%s: %s (%.4f); Word count: %s (%.4f)" % (label, count, count/N, word_count, word_count/np.sum([living_count, factory_count])))

"""**Question: What do we observe from the distribution? :)**

Let's take some time to discuss some assumptions.

This obviously doesn't tell us much, but it does give us an overview of what to expect from out data. But with that, we've successfully loaded our dataset and have a brief overview of it.

# 3. Cleaning our raw text data (Pre-processing)

**Data cleaning** is an important task in NLP as in most cases, we have horriblely structured data. Fortunately, we actually don't need to clearn our data too much since its actually quite organized. We do just need to remove some punctuations. So let's get on modifying our $X$. Since we're just removing punctuation, we can use reuse what we did awhile ago using Python's string translate.
"""

punct = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{}~'
transtab = str.maketrans(dict.fromkeys(punct, ''))

print(X[6])

i = 0
while i < len(X):
  X[i] = X[i].translate(transtab)
  i += 1
  
print(X[6])

"""Kindly note the removal of punctuation like this does have good and bad effects on the entire corpus. Good where we get rid of unwanted features, but bad where worods like `Ryburn's` turns to `Ryburn`. Do note that we're assuming we'll lose some information here and we just need to be ready to justify it.

Cleaning of data might even refer to using lexicons to categorize words, applying POS tagging, looking at higher n-grams, etc. The basic idea is to ready the data for processing - hence why we call it **Data Pre-processing**!

# 4. Splitting data into training and testing sets

When looking to perform machine learning, we need a way to evaluate the performance of our models, so we look to split out data into training and testing sets. So here's the idea:

![alt text](https://cdn-images-1.medium.com/max/1600/1*-8_kogvwmL1H6ooN1A1tsQ.png =400x)

*source: https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6*

Our **training set** becomes the knowledge we learn, but it is possible to overfit on our knowledge - or memorize the features. The problem with memorization is that its hard to generalize unless its an exact match. So to test if our learned knowledge is generalizable, we get data that we have not seen / have not learned from and test - hence, the **test set**! There are also other concepts needed to be learned for the ML side, but we'll only tackle the bare minimum.

So on to some code! Let's first split our data into training and test. We'll set our testing size to 30% of our total dataset, leaving 70% for training. These numbers can be changed around, but for this session, these numbers should be good.
"""

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, # Features
    Y, # Labels
    test_size = 0.3, # The defined test size; Training size is just 1 minus the test size
    random_state = 12 # So we can shuffle the elements, but with some consistency
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

"""By default `train_test_split` performs a stratified split based on the classes.
Stratify just means to arrange/sort into groups, so with our data, it balances
our the 2 classes (living versus factory). 

See more about the function here: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

**Question: What would happen if we just randomly got indexes from the whole dataset? What implications could happen if we performed stratification or not?**

# 5. Extracting features from text

We discussed two ways to extract information in class: collocation and bag-of-words. Collocation looks at the words $\pm N$ from an anchor word at index $i$. We can typically look at the words and their POS tag. For bag-of-words, we don't care about the order and look at what words are found in the document. Both representations can be shown in the form of a vector, so in this session, we'll just focus on extracting a bag-of-words model. Let's look at sklearn's `CountVectorizer`

**Remember! Don't touch the testing data yet!**
"""

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorizer

"""Some parameters to note:


*   Analyzer - We can analyze at the **word** or **char**(acter) level
*   Binary - If binary is set to true, it will count if the word appeared or not (0 or 1). If set to false, it will count how many times the word appeared.
*   Max_df & min_df - Maximum and minimum document frequency. If a word is too common or too unique with respect to appearing across documents, it can cause noise. This can be solved by setting the min and max document frequencies.
*   Max_features - Considers the top max_features ordered by term frequency
*   Ngram_range - (1, 1) means we're only looking at 1-grams. (1, 2) means include 1 and 2-grams. Think of it as min and max.
*   Tokenizer - you can set this as your own function for tokenizing!
*   Vocabulary - You can set your own vocabulary here. If not, then it'll consider all words.

For more info, check out their site: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer

So let's trying running it on our training data.
"""

import pandas as pd

x = vectorizer.fit_transform(X_train)
count_vect_df = pd.DataFrame(x.todense(), columns=vectorizer.get_feature_names())

count_vect_df.head()

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
    X, # Our training set
    Y, # Our test set
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
  f1 = f1_score(Y_test, y_pred, pos_label='factory_sense')  
  
  print("%s\nAccuracy: %s\nF1 Score: %s\n=====" % (clf, acc, f1))

"""Look! The Naive Bayes implementation performed best! This means that we're able to correctly predict up to 80.7% accuracy whether a given sentence that uses plant as either the living sense or the factory sense. Not bad, but not good.

**HOWEVER**, this performance is because we didn't perform cross-validation and actually tune our models. These are all things that will be covered outside of NLP, but are very important to gain a grasp of classification, in general. Our models have many parameters, so we need to tune them to our training data and hope they can generalize when we test our the testing set. 

But at least for now, we've performed the bare minimum for text classification. We've loaded the corpus in, extracted information, learned from the model, and tested it out on unseen data.

The assignment will be uploaded via Canvas, but you'll make sure of this Python notebook in exploring the effects of pre-processing, feature extraction, and classification.

# 7. Memes of the week

Cause you all are awesome, here are the memes of the week

![alt text](https://scontent.fmnl10-1.fna.fbcdn.net/v/t1.15752-9/52458975_2541773996049221_2083917586558353408_n.png?_nc_cat=100&_nc_ht=scontent.fmnl10-1.fna&oh=b484b4da148a84fe6a039e0d7cc97e27&oe=5CF6C184)

Meme submitted by Blaise Cruz (yeaboi)

![alt text](https://scontent.fmnl10-1.fna.fbcdn.net/v/t1.0-9/49209747_520866785074567_581292375763058688_n.png?_nc_cat=101&_nc_ht=scontent.fmnl10-1.fna&oh=005e4c0848f79f80bab633c142da65f4&oe=5CDE8ADE)
"""