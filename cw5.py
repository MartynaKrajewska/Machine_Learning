# -*- coding: utf-8 -*-
# Przykład oparty na kodzie z: 
# http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html
# Authors: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD 3 clause
# adaptacja: Jarosław Żygierewicz
 
 
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import  MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
 
# W całym zbiorze danych jest 20 list dyskusyjnych, tu wykorzystamy podzbiór:
# kategorie dla których zbudujemy klasyfikator
categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space'  ]
 
 
# Ładujemy dane z newsgroups dataset dla wybranch kategorii
# korzystamy z funkcji sklearn.datasets.fetch_20newsgroups
# http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html#sklearn.datasets.fetch_20newsgroups
data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=('headers', 'footers', 'quotes'))
 
data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=('headers', 'footers', 'quotes'))
 
categories = data_train.target_names 
 
# zobaczmy jak wyglądają przykładowe dane
id =57
print( data_train.data[id])          # lista wiadomości
print (data_train.target[id])       # lista kodów tematycznych
print (categories[data_train.target[id]]) # nazwy kategorii odpowiadających kodom
 
# upraszczamy nazewnictwo
y_train, y_test = data_train.target, data_test.target
 
 
# przekodowujemy wiadomości na wekotry cech 
# korzystamy z funkcji: sklearn.feature_extraction.text.TfidfVectorizer 
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer

vectorizer = TfidfVectorizer() # stwórz instancje obiektu TfidfVectorizer


vectorizer.fit(data_train.data)# naucz vctorizer słownika i przetransformuj dane uczące.  
X_train = vectorizer.transform(data_train.data)


# wypisz rozmiary danych treningowych
print("Dane treningowe: n_samples: %d, n_features: %d" % X_train.shape)
# Dane uczące są przechowywane w macierzy rzadkiej (sparse matrix)
# proszę podejrzeć jak wyglądają tak przekodowane dane:
print (X_train)
print
 


X_test = vectorizer.transform(data_test.data) # wektoryzujemy też dane testowe
print("Dane testowe: n_samples: %d, n_features: %d" % X_test.shape)
print(X_test)
print
 
# odwrotne mapowanie z cech na słowa
feature_names = vectorizer.get_feature_names()
feature_names = np.asarray(feature_names)



# tworzymy instancję i uczymy klasyfikator MultinomialNB
clf = MultinomialNB()
clf.fit( X_train, y_train)

# Benchmark: tu będziemy korzystać z funkcji zaimplementowanych w 
# http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
 
pred = clf.predict(X_test) # obliczamy predykcję dla tekstów ze zbioru testowego
accur = accuracy_score(y_test, pred) #dokladnosc
print("dokladnosc:   %0.3f" % accur)

print("classification report:") # wypisz raport klasyfikacji 
print(classification_report(y_test, pred))
 
print("Macierz błędów") # wypisz macierz (confusion matrix)
print(confusion_matrix(y_test, pred))

# wypiszemy teraz po 10 najbardziej znaczących słów w każdej klasie
print("top 10 keywords per class:")
for i, category in enumerate(categories):
    top10 = np.argsort(clf.coef_[i])[-10:]
    print("%s: %s" % (category, " ".join(feature_names[top10])))
print

