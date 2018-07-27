# -*- coding: utf-8 -*-
"""
Created on Wed May 10 08:38:21 2017

@author: ml383488
"""

from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

# doinstaluj pydot

## bibloteka Graphiz
#from sklearn.externals.six import StringIO  
#import pydot 
#dot_data = StringIO() 
#tree.export_graphviz(clf, out_file=dot_data) 
#graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
#graph.write_pdf("iris.pdf")

### wyniki w pythonie
#from IPython.display import Image  
#dot_data = StringIO()  
#tree.export_graphviz(clf, out_file=dot_data,    feature_names=iris.feature_names,  
#                         class_names=iris.target_names,  
#                         filled=True, rounded=True,  
#                         special_characters=True) 
#graph = pydot.graph_from_dot_data(dot_data.getvalue())  
#Image(graph.create_png())

clf.predict(iris.data[:1, :])
clf.predict_proba(iris.data[:1, :])

