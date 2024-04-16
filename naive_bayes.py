#-------------------------------------------------------------------------
# AUTHOR: Dhruvi Choksi
# FILENAME: naive_bayes.py
# SPECIFICATION: description of the program
# FOR: CS 5990- Assignment #3
# TIME SPENT: 1-2 hours
#-----------------------------------------------------------*/

#importing some Python libraries
import pandas as pd
from sklearn.naive_bayes import GaussianNB

# 11 classes after discretization
classes = [i for i in range(-22, 40, 6)]

# Reading the training data
weather_training = pd.read_csv('weather_training.csv')
y_trainingset = weather_training['Temperature (C)'].values.astype('float')
x_trainingset = weather_training.drop(columns=['Formatted Date', 'Temperature (C)']).values.astype('float')

# Discretize the target variable according to the discretization classes
y_trainingset_discretized = pd.cut(y_trainingset, bins=classes, labels=False)

# Reading the test data
weather_test = pd.read_csv('weather_test.csv')
x_test = weather_test.drop(columns=['Formatted Date', 'Temperature (C)']).values.astype('float')
y_test = weather_test['Temperature (C)'].values.astype('float')

# Fitting the Naive Bayes classifier to the training data
clf = GaussianNB()
clf = clf.fit(x_trainingset, y_trainingset_discretized)

# Make predictions on the test data
y_pred = clf.predict(x_test)

# Calculate accuracy
correct_predictions = sum(0.85 * y_test <= y_pred <= 1.15 * y_test)
accuracy = correct_predictions / len(y_test)

# Print the accuracy
print("Naive Bayes accuracy:", accuracy)


