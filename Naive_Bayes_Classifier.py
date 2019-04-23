# Importing required libraries
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet as wn

from stop_words import get_stop_words

import re

from sklearn import preprocessing, model_selection, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Importing required libraries
import pandas as pd
​
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet as wn
​
from stop_words import get_stop_words
​
import re
​
from sklearn import preprocessing, model_selection, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

df = pd.read_csv("/home/sneha/Documents/Text_Classification/Consumer_Complaints.csv")

df.head(10)

# Labels to classify into
df.Product.unique()
df = df[["Consumer complaint narrative", "Product"]]
df.dropna(subset = ['Consumer complaint narrative'], how='all', inplace = True)
len(df)
df.head(10)

# Train - Test Split
train_x, test_x, train_y, test_y = model_selection.train_test_split(df['Consumer complaint narrative'], df['Product'])

# Label encoding - encode the labels by assigning numerical value to each class
label_encoder = preprocessing.LabelEncoder()
train_y = label_encoder.fit_transform(train_y)
test_y = label_encoder.fit_transform(test_y)

# Converting texts to a matrix of token counts
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(df['Consumer complaint narrative'])

# Procuring texts-terms matrix (w1,w2) - count
train_count =  count_vect.transform(train_x)
test_count =  count_vect.transform(test_x)

# Fit the model, predict the labels and measure accuracy
fit_naive_bayes = naive_bayes.MultinomialNB().fit(train_count, train_y)
predictions = fit_naive_bayes.predict(test_count)
metrics.accuracy_score(predictions, test_y)

# Alternative
fit_naive_bayes.score(test_count,test_y)

# Converting texts into TF-IDF feature
tfidf_converter = TfidfVectorizer(max_features=5000, norm='l2', stop_words='english')
tfidf_converter.fit(df['Consumer complaint narrative'])
train_tfidf =  tfidf_converter.transform(train_x)
test_tfidf =  tfidf_converter.transform(test_x)

# Fit the model, predict the labels and measure accuracy
fit_naive_bayes = naive_bayes.MultinomialNB().fit(train_tfidf, train_y)
predictions = fit_naive_bayes.predict(test_tfidf)
metrics.accuracy_score(predictions, test_y)


