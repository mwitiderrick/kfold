# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd 
df = pd.read_csv('imdb_labelled.txt', delimiter = '\t', quoting = 3)

#cleaning the dataset
import re 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
#Create BOW model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfVectorizer = TfidfVectorizer(max_features = 1500)
X = tfidfVectorizer.fit_transform(corpus).toarray()
y = df.iloc[:, 1].values

from sklearn.externals import joblib
joblib.dump(tfidfVectorizer, 'tfidfVectorizer.pkl')


from sklearn.model_selection import RepeatedKFold
rkf = RepeatedKFold(n_splits=10, n_repeats=2)
for train_index, test_index in rkf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


#Fit Naive Bayes to the training set 
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train) 
joblib.dump(classifier, 'classifier.pkl')
 
#Predicting the test set results
y_pred = classifier.predict(X_test)

#Making the confusion matrix 
from sklearn.metrics import confusion_matrix, classification_report, r2_score
cm = confusion_matrix(y_test, y_pred) 
cr = classification_report(y_test,y_pred)



