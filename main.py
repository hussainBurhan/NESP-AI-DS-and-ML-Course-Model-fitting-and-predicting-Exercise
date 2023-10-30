# Importing necessary libraries
import pandas as pd

# Loading the dataset
meternal_health_risk = pd.read_csv('Maternal Health Risk Data Set.csv')

# Displaying information about the dataset
meternal_health_risk.info()

# Converting 'RiskLevel' to binary labels (0 for low risk, 1 for high risk)
meternal_health_risk['RiskLevel'] = meternal_health_risk['RiskLevel'].str.replace('high risk', '1')
meternal_health_risk['RiskLevel'] = meternal_health_risk['RiskLevel'].str.replace('mid risk', '0')
meternal_health_risk['RiskLevel'] = meternal_health_risk['RiskLevel'].str.replace('low risk', '0').astype(int)

# Separating features (x) and target variable (y)
x = meternal_health_risk.drop('RiskLevel', axis= 1)
y = meternal_health_risk['RiskLevel']

# Importing the RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# Importing train_test_split for splitting the dataset
from sklearn.model_selection import train_test_split

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Initializing and training the RandomForestClassifier
clf = RandomForestClassifier().fit(x_train, y_train)

# Predicting the target variable
y_predicted = clf.predict(x_test)

# predicting probability
predicted_proba = clf.predict_proba(x_test)[:5]
print('predicted probability:')
print(predicted_proba)

# Printing the predicted values
print('y_predicted:')
print(y_predicted)

# Importing numpy for mathematical operations
import numpy as np

# Calculating accuracy using Method 1 (numpy)
print('Method 1 accuracy')
print(np.mean(y_predicted == y_test))

# Calculating accuracy using Method 2 (clf.score)
print('Method 2 accuracy')
print(clf.score(x_test, y_test))

# Importing accuracy_score from sklearn for accuracy calculation
from sklearn.metrics import accuracy_score

# Calculating accuracy using Method 2 (accuracy_score)
print('Method 2 accuracy')
print(accuracy_score(y_test, y_predicted))
