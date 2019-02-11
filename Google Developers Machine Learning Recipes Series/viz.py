#1 Import Data 
import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()

#Test dataset prints 
'''
print(iris.feature_names)
print(iris.target_names)
print(iris.data[0])
print(iris.target[0])
'''

#Print Entire Dataset 
'''
for i in range(len(iris.target)):
    print("Example %d: label %s, features %s" % (i, iris.target[i], iris.data[i]))
'''

#2 Data Testing & Training

test_idx = [0,50,100]

#split data
#training data
train_target = np.delete(iris.target, test_idx)
#print(train_target)
train_data = np.delete(iris.data, test_idx, axis = 0)
#print(train_data)

#testdata
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier() #clasifir
clf.fit(train_data,train_target)


#3 Predict label for a new flower 
#print(train_target)
#See the value of what we removed [0 1 2]
print(test_target)
#This should match what was removed if predicted correctly
print(clf.predict(test_data))
#It does!

print(test_data[0], train_target[0])

#Visualize the Decision Tree
#This will create a iris.pdf when complete

import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=iris.feature_names,  
                     class_names=iris.target_names,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.render("iris")  

#Testing some predictions looking at the Visualized Tree 
print(test_data[0], test_target[0])
print(test_data[1], test_target[0])
#Questions go from Right to Left when making decisions
print(iris.feature_names, iris.target_names)

