import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df = pd.read_table("Question2 Dataset.tsv")
df = df.drop(columns="id")

for index, row in df.iterrows():
    soup = BeautifulSoup(row[1], "html.parser")
    text = soup.get_text()
    text = text.replace('\\', '')
    df.at[index, 'review'] = text

vectorizer = CountVectorizer()
transformer = TfidfTransformer()
# calculate Raw Counts
rawCount = vectorizer.fit_transform(df['review'])
# calculate IDF
tfIDF = transformer.fit(rawCount)
# calculate TfIDF
tfIDF = transformer.transform(rawCount)

X_train, X_test, y_train, y_test = train_test_split(rawCount, df['sentiment'], test_size=0.2, random_state=1999)
rawCountModel = MultinomialNB().fit(X_train, y_train)
rawCountPredicted = rawCountModel.predict(X_test)

X_train, X_test = train_test_split(tfIDF, test_size=0.2, random_state=1999)
tfIDFModel = MultinomialNB().fit(X_train, y_train)
tfIDFPredicted = tfIDFModel.predict(X_test)

print("Raw Counts Accuracy:\t" + str(np.mean(rawCountPredicted == y_test)*100))
print("TfIDF Accuracy:\t\t" + str(np.mean(tfIDFPredicted == y_test)*100))
