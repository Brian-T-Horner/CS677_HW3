"""
Brian Horner
CS 677 - Summer 2
Date: 7/27/2021
Week 3 Homework Question 4
This program uses k-NN k=3 and drops features one at a time. We calculate
accuracy each time we drop a feature and compare it to the original k-NN k=3.
"""

# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

    # Creating training data with each feature dropped
    f1_dropped_train = np.delete(X, obj=0, axis=1)
    f2_dropped_train = np.delete(X, obj=1, axis=1)
    f3_dropped_train = np.delete(X, obj=2, axis=1)
    f4_dropped_train = np.delete(X, obj=3, axis=1)

    # Creating test data with each feature dropped
    f1_dropped_test = np.delete(test_data, obj=0, axis=1)
    f2_dropped_test = np.delete(test_data, obj=1, axis=1)
    f3_dropped_test = np.delete(test_data, obj=2, axis=1)
    f4_dropped_test = np.delete(test_data, obj=3, axis=1)

    # k-NN k=3 for feature 1 dropped
    f1_dropped_knn_3 = KNeighborsClassifier(n_neighbors=3)
    f1_dropped_knn_3.fit(f1_dropped_train, Y)
    f1_dropped_knn_predicted = f1_dropped_knn_3.predict(f1_dropped_test)
    f1_dropped_acc = round(accuracy_score(test_values,
                                          f1_dropped_knn_predicted), 3)

     # k-NN k=3 for feature 2 dropped
    f2_dropped_knn_3 = KNeighborsClassifier(n_neighbors=3)
    f2_dropped_knn_3.fit(f2_dropped_train, Y)
    f2_dropped_knn_predicted = f2_dropped_knn_3.predict(f2_dropped_test)
    f2_dropped_acc = round(accuracy_score(test_values,
                                          f2_dropped_knn_predicted), 3)

     # k-NN k=3 for feature 3 dropped
    f3_dropped_knn_3 = KNeighborsClassifier(n_neighbors=3)
    f3_dropped_knn_3.fit(f3_dropped_train, Y)
    f3_dropped_knn_predicted = f3_dropped_knn_3.predict(f3_dropped_test)
    f3_dropped_acc = round(accuracy_score(test_values,
                                          f3_dropped_knn_predicted), 3)

     # k-NN k=3 for feature 4 dropped
    f4_dropped_knn_3 = KNeighborsClassifier(n_neighbors=3)
    f4_dropped_knn_3.fit(f4_dropped_train, Y)
    f4_dropped_knn_predicted = f4_dropped_knn_3.predict(f4_dropped_test)
    f4_dropped_acc = round(accuracy_score(test_values,
                                          f4_dropped_knn_predicted), 3)


    """Question 4.1"""
    print('\t')
    print("--- Question 4.1 ---")
    print(f"Feature 1 dropped k-NN k=3 accuracy score {f1_dropped_acc}.")
    print(f"Feature 2 dropped k-NN k=3 accuracy score {f2_dropped_acc}.")
    print(f"Feature 3 dropped k-NN k=3 accuracy score {f3_dropped_acc}.")
    print(f"Feature 4 dropped k-NN k=3 accuracy score {f4_dropped_acc}.")
    print(formatter)

    """Question 4.2"""
    print('\t')
    print("--- Question 4.2 ---")
    print("None of the accuracies of the k-NN model were increased with a "
          "feature removed compared to the accuracy of all features which "
          "resulted in 0.99.")
    print(formatter)


    """Question 4.3"""
    print('\t')
    print("--- Question 4.3 ---")
    print("When the variance was removed from the testing and training data "
          "the loss of accuracy was the greatest at around a loss of 0.06.")
    print(formatter)


    """Question 4.4"""
    print('\t')
    print("--- Question 4.4 ---")
    print("When the entropy of the image was removed the loss of accuracy was "
          "the least with a loss around .002.")
    print(formatter)


