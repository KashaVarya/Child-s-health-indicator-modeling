from sklearn import datasets
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from pandas_ml import ConfusionMatrix

newsgroup_train = datasets.fetch_20newsgroups(subset='train')
newsgroup_test = datasets.fetch_20newsgroups(subset='test')

print(newsgroup_train.keys())
print(newsgroup_train.data[:3])
print(newsgroup_train.target[:3])
print(newsgroup_train.target_names)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(newsgroup_train.data)
X_test = vectorizer.transform(newsgroup_test.data)

y_train = newsgroup_train.target
y_test = newsgroup_test.target

model = MultinomialNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(model.score(X_test, y_test))
print(metrics.classification_report(y_test, predictions))
print(metrics.confusion_matrix(y_test, predictions))

labels = list(newsgroup_train.target_names)
cm = ConfusionMatrix(y_test, predictions, labels)
cm.plot()
plt.show()

