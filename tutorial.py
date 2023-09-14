import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

cancer = datasets.load_breast_cancer()  #importing the datasets from the library

#dividing the datasets into input features(x) and target labels(y)

x = cancer.data
y = cancer.target

#splitting the data into training and testing sets
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

classes = ['malignant', 'benign']

clf = svm.SVC(kernel="linear", C=2) #creating a SVM classifier

clf.fit(x_train,y_train) #training the SVM classifier

y_pred = clf.predict(x_test)  #Predictions are made on the test data using the trained classifier

acc = metrics.accuracy_score(y_test, y_pred)  #calculating the accuracy of the classifier
print("Accuracy: ",acc)

