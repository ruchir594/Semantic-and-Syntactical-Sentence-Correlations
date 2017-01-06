from sklearn.feature_extraction.text import TfidfVectorizer
corpus = ["The game of life is a game of everlasting learning",
          "The unexamined life is not worth living",
          "Never stop learning"]
vectorizer = TfidfVectorizer(min_df=1)
X = vectorizer.fit_transform(corpus)
idf = vectorizer.idf_
print dict(zip(vectorizer.get_feature_names(), idf))

corpus = ["life learning"]
X = vectorizer.fit_transform(corpus)
idf = vectorizer.idf_
print dict(zip(vectorizer.get_feature_names(), idf))

import json
hashing = dict()

with open('hashing.json', 'w') as fp:
    json.dump(hashing, fp)
