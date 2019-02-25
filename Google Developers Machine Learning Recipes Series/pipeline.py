#import a dataset
from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .5)

#using Decision Tree Classifier
#from sklearn import tree
#my_classifier = tree.DecisionTreeClassifier()

using K Nearest Neighbors Classifier 
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()



#If you want to use a better classifier you only need to change these two lines ^
#Takeaway: even though there are many different types of classifiers, at a high level they have a similar interface. 

my_classifier.fit(X_train, y_train) 

predictions = my_classifier.predict(X_test)
#print(predictions)

from sklearn.metrics import accuracy_score
print( accuracy_score(y_test, predictions))

