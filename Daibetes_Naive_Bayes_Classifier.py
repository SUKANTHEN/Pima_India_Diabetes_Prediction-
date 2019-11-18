import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
data = pd.read_csv("/home/admin1/DataFlair/Data/diabetes.csv")
X = data.iloc[:, 0:8].values
y = data.iloc[:, 8].values
data.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = GaussianNB() 
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
print(y_predict)
predictions3 = [np.round(value) for value in y_predict]
accuracy = accuracy_score(y_test, predictions3)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Now check for any women by providing any one-dimensional array values of all 8 clinical parameters
X_test = np.array([2 , 150, 72, 32, 0, 23, 0.16, 52])
Y_predict = clf.predict(X_test, reshape(1,8))
Y_predict
