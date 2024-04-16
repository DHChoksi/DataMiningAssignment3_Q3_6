#-------------------------------------------------------------------------
# AUTHOR: Dhruvi Choksi
# FILENAME: knn.py
# SPECIFICATION: knn with grid search
# FOR: CS 5990- Assignment #3
# TIME SPENT: 1-2 hours
#-----------------------------------------------------------*/

#importing some Python libraries
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

#11 classes after discretization
classes = [i for i in range(-22, 40, 6)]

#defining the hyperparameter values of KNN
k_values = [i for i in range(1, 20)]
p_values = [1, 2]
w_values = ['uniform', 'distance']

#reading the training data
weather_training = pd.read_csv('weather_training.csv')
y_trainingset = weather_training['Temperature (C)'].values.astype('float')
x_trainingset = weather_training.drop(columns=['Formatted Date', 'Temperature (C)']).values.astype('float')

#reading the test data
weather_test = pd.read_csv('weather_test.csv')
x_test = weather_test.drop(columns=['Formatted Date', 'Temperature (C)']).values.astype('float')
y_test = weather_test['Temperature (C)'].values.astype('float')

#hint: to convert values to float while reading them -> np.array(df.values)[:,-1].astype('f')
highest_accuracy = 0.0
best_parameters = {}

#loop over the hyperparameter values (k, p, and w) ok KNN
for k in k_values:
    for p in p_values:
        for w in w_values:

            #fitting the knn to the data
            clf = KNeighborsRegressor(n_neighbors=k, p=p, weights=w)
            clf.fit(x_trainingset, y_trainingset)

            #make the KNN prediction for each test sample and start computing its accuracy
            right_prediction = 0
            total_samples = len(x_test)
            for i in range(total_samples):
                predicted_value = clf.predict([x_test[i]])[0]
                real_value = y_test[i]
                difference_percentage = 100 * abs(predicted_value - real_value) / real_value
            #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values.
                if (real_value - 0.15 * real_value) <= predicted_value <= (real_value + 0.15 * real_value):
                    right_prediction += 1

            #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
            accuracy = right_prediction / total_samples
            #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                best_parameters = {'k': k, 'p': p, 'weight': w}

print(f'Highest accuracy: {highest_accuracy:0.2f}, Parameters: k={best_parameters["k"]}, p={best_parameters["p"]}, weight={best_parameters["weight"]}')




