"""
Brian Horner
CS 677 - Summer 2
Date: 7/27/2021
Week 3 Homework Question 5
This program uses logistic regression model on the data. We then compare it
to the simple classifier and k-NN k=3.
"""

# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":

    formatter = '--------------------------------------------------------------'
    # Reading data as a csv and adding columns
    df = pd.read_csv('data_banknote_authentication.txt', header=None)
    df.columns = ['Variance', 'Skewness', 'Curtosis', 'Entropy', 'Class']

    # Adding 'Color' column
    df['Color'] = np.where(df['Class'] == 0, 'Green', 'Red')

    X_train, X_test = train_test_split(df, test_size=0.5,
                                       random_state=33903)
    # Scaling data` ``          `   `       ``````
    scaler = StandardScaler()
    test_data = X_test[['Variance', 'Skewness', 'Curtosis', 'Entropy']].values
    scaler.fit(test_data)
    test_data = scaler.transform(test_data)
    test_values = X_test['Class'].values

    # Getting X and Y data from training set for Logistic Regression
    X = X_train.iloc[:, :-2].values
    scaler.fit(X)
    X = scaler.transform(X)
    Y = X_train.iloc[:, 4].values

    # Using Logistic Regression
    log_reg_classifier = LogisticRegression()
    log_reg_classifier.fit(X, Y)

    # Predicting with Logistic Regression
    predicted = log_reg_classifier.predict(test_data)

    # Using confusion matrix for tp, fn, fp, tn
    conf_matrix = confusion_matrix(test_values, predicted)
    tp = conf_matrix[0][0]
    fn = conf_matrix[0][1]
    fp = conf_matrix[1][0]
    tn = conf_matrix[1][1]

    # Calculating TPR, TNR, and Accuracy
    TPR = round(precision_score(test_values, predicted), 4)
    TNR = round((conf_matrix[1][1])/(conf_matrix[1][1] + conf_matrix[1][0]), 4)
    accuracy = round(accuracy_score(test_values, predicted), 4)

    # Table print code for statistics below
    header_list = ['TP', 'FP', 'TN', 'FN', 'Accuracy', 'TPR', 'TNR']
    stats_list = [tp, fp, tn, fn, accuracy, TPR, TNR]
    print_list = []
    print_list.append(header_list)
    print_list.append(stats_list)
    print("--- Question 5.2 ---\n")
    print("--- Logistic Regression Statistics ---")
    for index, stuff in enumerate(print_list):
        # Adding a | in front of each value of the lists in print list
        row = '|'.join(str(value).ljust(10) for value in stuff)
        # Printing the row for the list in print list
        print(row)
        # Adding a line between the header and the data rows
        if index == 0:
            print('-' * len(row))


    """Question 5.3"""
    print("\n --- Question 5.3 ---")
    print("The logistic regression is better than my simple classifiers in "
          "all metrics computed.")
    """Question 5.4"""
    print("\n--- Question 5.4 ---")
    print("No the logistic regression is worse than the k-NN k=3 in all "
          "accounts except for the True Negative Rate in which it is equal.")
    print(formatter)

    """Question 5.5"""
    print('\t')
    print("--- Question 5.5 ---")
    student_id = np.array([2, 8, 1, 4])
    student_id = student_id.reshape(1, -1)
    print(f"With the last found of my BUID (2,8,1,4) the k-NN k"
          f"=3 predicts that it is a valid bank note "
          f"({log_reg_classifier.predict(student_id)[0]}).")
    print(formatter)
