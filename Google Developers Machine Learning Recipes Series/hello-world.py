from sklearn import tree
#features 
#0 -> bumpy
#1 -> smooth
#labels 
#0 -> apple
#1 -> orange
features = [[140,1],[130,1],[150,0],[170,0]]
labels = [0,0,1,1]
#import tree and classifier 
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

#Test with a new data point, Should return 1 for Orange 
print(clf.predict([[160,0]]))