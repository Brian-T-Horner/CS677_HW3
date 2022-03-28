"""
Brian Horner
CS 677 - Summer 2
Date: 7/27/2021
Week 3 Homework Question 3
This program applies k-NN k = 3, 5, 7, 9, and 11. We plot the accuracies of
all and decide which is the best. We then calculate statistics for the best k value.
"""

# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

if __name__ == "__main__":

    formatter = '--------------------------------------------------------------'
    """Question 1.1"""
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

    print('\n')
    # Using k-NN k=3
    knn_3 = KNeighborsClassifier(n_neighbors=3)
    knn_3.fit(X, Y)
    knn_3_acc = knn_3.score(X, Y)
    knn_3_y_predict = knn_3.predict(test_data)
    print(f" Accuracy score of k-NN 3 is {accuracy_score(test_values,knn_3_y_predict)}.")

    # Using k-NN k=5
    knn_5 = KNeighborsClassifier(n_neighbors=5)
    knn_5.fit(X, Y)
    knn_5_acc = knn_5.score(X, Y)
    knn_5_y_predict = knn_5.predict(test_data)
    print(f" Accuracy score of k-NN 5 is"
          f" {accuracy_score(test_values,knn_5_y_predict)}.")

    # Using k-NN k=7
    knn_7 = KNeighborsClassifier(n_neighbors=7)
    knn_7.fit(X, Y)
    knn_7_acc = knn_7.score(X, Y)
    knn_7_y_predict = knn_7.predict(test_data)
    print(f" Accuracy score of k-NN 7 is"
          f" {accuracy_score(test_values,knn_7_y_predict)}.")

    # Using k-NN k=9
    knn_9 = KNeighborsClassifier(n_neighbors=9)
    knn_9.fit(X, Y)
    knn_9_acc = knn_9.score(X, Y)
    knn_9_y_predict = knn_9.predict(test_data)
    print(f" Accuracy score of k-NN 9 is"
          f" {accuracy_score(test_values,knn_9_y_predict)}.")

    # Using k-NN k=11
    knn_11 = KNeighborsClassifier(n_neighbors=11)
    knn_11.fit(X, Y)
    knn_11_acc = knn_11.score(X, Y)
    knn_11_y_predict = knn_11.predict(test_data)
    print(f" Accuracy score of k-NN 11 is"
          f" {accuracy_score(test_values,knn_11_y_predict)}.")
    print("\n")
    knn_list = [3, 5, 7, 9, 11]

    # Calculating knn value accuracys
    knn_3_acc = accuracy_score(test_values,knn_3_y_predict)
    knn_5_acc = accuracy_score(test_values,knn_5_y_predict)
    knn_7_acc = accuracy_score(test_values,knn_7_y_predict)
    knn_9_acc = accuracy_score(test_values,knn_9_y_predict)
    knn_11_acc = accuracy_score(test_values,knn_11_y_predict)
    accuracy_list = []
    accuracy_list.append(knn_3_acc)
    accuracy_list.append(knn_5_acc)
    accuracy_list.append(knn_7_acc)
    accuracy_list.append(knn_9_acc)
    accuracy_list.append(knn_11_acc)

    """Question 3.2"""
    # Making a scatter plot of k values and their accuracies
    plt.scatter([3, 5, 7, 9, 11],
            [knn_3_acc, knn_5_acc, knn_7_acc, knn_9_acc, knn_11_acc],
             )
    plt.ylabel('k-NN k')
    plt.xlabel('k-NN Accuracy Score')






    """Question 3.3"""
    print("--- Question 3.3 --- ")
    # Using confusion matrix for tp, fn, fp and tn
    conf_matrix = confusion_matrix(test_values, knn_3_y_predict)
    tp = conf_matrix[0][0]
    fn = conf_matrix[0][1]
    fp = conf_matrix[1][0]
    tn = conf_matrix[1][1]
    # Calculating TPR, TNR and Accuracy
    TPR = round(precision_score(test_values, knn_3_y_predict), 2)
    TNR = round((conf_matrix[1][1])/(conf_matrix[1][1] + conf_matrix[1][0]), 2)
    accuracy = round(knn_3_acc, 4)
    header_list = ['TP', 'FP', 'TN', 'FN', 'Accuracy', 'TPR', 'TNR']
    stats_list = [tp, fp, tn, fn, accuracy, TPR, TNR]
    print_list = []
    print_list.append(header_list)
    print_list.append(stats_list)
    print("-- k-NN k=3 Statistics ---")
    # Printing print list to table
    for index, stuff in enumerate(print_list):
        # Adding a | in front of each value of the lists in print list
        row = '|'.join(str(value).ljust(10) for value in stuff)
        # Printing the row for the list in print list
        print(row)
        # Adding a line between the header and the data rows
        if index == 0:
            print('-' * len(row))


    """Question 3.4"""
    print("\n")
    print("--- Question 3.4 ---")
    print("My best k-NN with k =3 has a greater accuracy than my simple "
          "classification by over 30%")

    """Question 3.5"""
    print("\n")
    print("--- Question 3.5 ---")
    student_id = np.array([2, 8, 1, 4])
    student_id = student_id.reshape(1, -1)
    print(f"With the last found of my BUID (2,8,1,4) the k-NN k"
          f"=3 predicts that it is a valid bank note "
          f"({knn_3.predict(student_id)[0]}).")

    # Printing graph down below so all output can be printed to terminal first
    plt.show()
