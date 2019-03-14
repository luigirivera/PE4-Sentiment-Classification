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