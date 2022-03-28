"""
Brian Horner
CS 677 - Summer 2
Date: 7/27/2021
Week 3 Homework Question 6
This program uses logistic regression and drops features one at a time. We
calculate accuracy each time we drop a feature. We compare it to the
original logistic regression.
"""

# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':

    formatter = ("------------------------------------------------------------")
    # Reading data as a csv and adding columns
    df = pd.read_csv('data_banknote_authentication.txt', header=None)
    df.columns = ['Variance', 'Skewness', 'Curtosis', 'Entropy', 'Class']

    # Adding 'Color' column
    df['Color'] = np.where(df['Class'] == 0, 'Green', 'Red')

    X_train, X_test = train_test_split(df, test_size=0.5,
                                       random_state=33903)
    # Scaling data
    scaler = StandardScaler()
    test_data = X_test[['Variance', 'Skewness', 'Curtosis', 'Entropy']].values
    scaler.fit(test_data)
    test_data = scaler.transform(test_data)
    test_values = X_test['Class'].values

    X = X_train.iloc[:, :-2].values
    scaler.fit(X)
    X = scaler.transform(X)
    Y = X_train.iloc[:, 4].values

    # Grabbing training data for each feature dropped
    f1_dropped_train = np.delete(X, obj=0, axis=1)
    f2_dropped_train = np.delete(X, obj=1, axis=1)
    f3_dropped_train = np.delete(X, obj=2, axis=1)
    f4_dropped_train = np.delete(X, obj=3, axis=1)

    # Grabbing testing data for each feature dropped
    f1_dropped_test = np.delete(test_data, obj=0, axis=1)
    f2_dropped_test = np.delete(test_data, obj=1, axis=1)
    f3_dropped_test = np.delete(test_data, obj=2, axis=1)
    f4_dropped_test = np.delete(test_data, obj=3, axis=1)

    # Using logistic regression one ach feature train and test set
    f1_log_reg_classifier = LogisticRegression()
    f1_log_reg_classifier.fit(f1_dropped_train, Y)

    f1_dropped_predicted = f1_log_reg_classifier.predict(f1_dropped_train)
    f1_dropped_acc = round(accuracy_score(test_values,
                                          f1_dropped_predicted), 3)

    f2_log_reg_classifier = LogisticRegression()
    f2_log_reg_classifier.fit(f2_dropped_train, Y)

    f2_dropped_predicted = f2_log_reg_classifier.predict(f2_dropped_train)
    f2_dropped_acc = round(accuracy_score(test_values,
                                          f2_dropped_predicted), 3)

    f3_log_reg_classifier = LogisticRegression()
    f3_log_reg_classifier.fit(f3_dropped_train, Y)

    f3_dropped_predicted = f3_log_reg_classifier.predict(f3_dropped_train)
    f3_dropped_acc = round(accuracy_score(test_values,
                                          f3_dropped_predicted), 3)

    f4_log_reg_classifier = LogisticRegression()
    f4_log_reg_classifier.fit(f4_dropped_train, Y)

    f4_dropped_predicted = f4_log_reg_classifier.predict(f4_dropped_train)
    f4_dropped_acc = round(accuracy_score(test_values,
                                          f4_dropped_predicted), 3)

    """Question 6.1"""
    print('\t')
    print("--- Question 6.1 ---")
    print(f"Feature 1 dropped logistic regression accuracy score"
          f" {f1_dropped_acc}.")
    print(f"Feature 2 dropped logistic regression accuracy score"
          f" {f2_dropped_acc}.")
    print(f"Feature 3 dropped logistic regression accuracy score"
          f" {f3_dropped_acc}.")
    print(f"Feature 4 dropped logistic regression accuracy score"
          f" {f4_dropped_acc}.")
    print(formatter)

    """Question 6.2"""
    print('\t')
    print("--- Question 6.2 ---")
    print(f" The accuracy decreased when removing any of the features from "
          f"the data.")
    print(formatter)

    """Question 6.3"""
    print('\t')
    print("--- Question 6.3 ---")
    print(f"The removal of Entropy contributed to the most loss of accuracy.")
    print(formatter)

    """Question 6.4"""
    print('\t')
    print("--- Question 6.4 ---")
    print(f"The removal of Variance contributed to the least loss of accuracy.")
    print(formatter)

    """Question 6.5"""
    print('\t')
    print("--- Question 6.5 ---")
    print(f"No the relative significance of the features is vastly greater in a "
          f"logistic regression model. The loss of accuracy is extreme when removing any feature.")
    print(formatter)
