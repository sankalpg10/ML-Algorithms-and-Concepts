# Enter your code here. Read input from STDIN. Print output to STDOUT
import nltk

# nltk.download("punkt")
import sys
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report

with open("data/training_data.txt", "r") as f:
    lines = f.readlines()

rows = lines[0]
training_data = []
for line in lines[1:]:
    row = line.split(" ")
    cat, text = int(row[0]), row[1:-1]
    training_data.append([text, cat])

training_data = pd.DataFrame(training_data, columns=["Text", "Category"])

# vectorize the text
vect = CountVectorizer()
vect.fit(training_data["Text"].apply(lambda x: " ".join(x)))
X = vect.transform(training_data["Text"].apply(lambda x: " ".join(x)))
x_train = pd.DataFrame(X.toarray(), columns=vect.get_feature_names_out())
# print(vect.vocabulary_)

# print(vect.get_feature_names_out())


y_train = training_data["Category"]
print(training_data["Category"].value_counts())


clf = MultinomialNB()
clf.fit(x_train, y_train)


test_samples = 3


for line in range(test_samples + 1):
    text = [input()]

    y_pred = clf.predict(vect.transform(text))
    print(f"y_pred for  {text} is {y_pred}")
