import random

#Create EU distance
from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)


class ScrappyKNN():
    #Fit method that does the training. 
    def fit(self, X_train, y_train):
        #To Start we will guess the label
        self.X_train = X_train
        self.y_train = y_train
    
    #This prediction got 0.33333 as a result because there is not a very good classifier 
    #Using a random point is a terrible classifier 
    # #Predict Method does the testing.
    # def predict(self,X_test):
    #     predictions = []
    #     for row in X_test:
    #         label = random.choice(self.y_train)
    #         predictions.append(label)
    #     return predictions

    #Using Euc distance as the classifier
    #Using this Classifier the prediction is in the 90th percentile!
    def predict(self,X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    #Keep track of closest point so far:
    def closest(self,row):
        best_dist= euc(row,self.X_train[0])
        best_index = 0

        #iterate over all distances 
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            #if a closer distance is found update variables 
            if dist < best_dist:
                best_dist = dist
                best_index = i
        #use the index to return the label for the closest training example
        return self.y_train[best_index]





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

#using K Nearest Neighbors Classifier 
#from sklearn.neighbors import KNeighborsClassifier
#my_classifier = KNeighborsClassifier()

#Start of Episode 5 creation of K Nearest Neighbors


my_classifier = ScrappyKNN()

#If you want to use a better classifier you only need to change these two lines ^
#Takeaway: even though there are many different types of classifiers, at a high level they have a similar interface. 

my_classifier.fit(X_train, y_train) 

predictions = my_classifier.predict(X_test)
#print(predictions)

from sklearn.metrics import accuracy_score
print( accuracy_score(y_test, predictions))

