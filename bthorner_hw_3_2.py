"""
Brian Horner
CS 677 - Summer 2
Date: 7/27/2021
Week 3 Homework Question 2
This program prints pair wise plots for good and bad bills, applies a simple
model to the testing data and prints statistics for this model.
"""

# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns

def table_printer(stats_list):
    """Formats the computaations for table printing."""
    header_list = ['TP', 'FP', 'TN', 'FN', 'Accuracy', 'TPR', 'TNR']
    print_list = []
    print("--- Model with 'Variance' >=1 & 'Skewness' >= -1 & 'Curtosis' <=7 "
          "qualifying as a real bill. ---\n")
    print_list.append(stats_list)
    print_list.insert(0, list(header_list))
    for index, stuff in enumerate(print_list):
        # Adding a | in front of each value of the lists in print list
        row = '|'.join(str(value).ljust(12) for value in stuff)
        # Printing the row for the list in print list
        print(row)
        # Adding a line between the header and the data rows
        if index == 0:
            print('-' * len(row))
if __name__ == "__main__":

    # Reading data as a csv and adding columns
    df = pd.read_csv('data_banknote_authentication.txt', header=None)
    df.columns = ['Variance', 'Skewness', 'Curtosis', 'Entropy', 'Class']

    # Adding 'Color' column
    df['Color'] = np.where(df['Class'] == 0, 'Green', 'Red')
    """Question 2.1"""
    print("--- Question 2.1 ---\n")
    X_train, X_test = train_test_split(df, test_size=0.5, random_state=33903)

    # Splitting the test class by true and fake bank notes
    truenote_test = X_train[X_train['Class'] == 0]
    del truenote_test['Class']
    fakenote_test = X_train[X_train['Class'] == 1]
    del fakenote_test['Class']
    # Making pair plots for true and fake bank notes
    true_plot = sns.pairplot(truenote_test)
    true_plot.savefig('good_bills.pdf')
    false_plot = sns.pairplot(fakenote_test)
    false_plot.savefig('fake_bills.pdf')

    print("---'Pairplots' saved under good_bils.pdf and fake_bills.pdf ---\n")
    """Question 2.3"""
    # Applying classification to test_data
    test_data = X_test[(X_test['Variance'] >=1.3) & (X_test['Curtosis']<=6.5
                                                   )
                         & (X_test['Skewness']>=-1.25)]
    index_list = test_data.index.tolist()
    label_list = []
    # Making list of True and False bank notes
    for index, row in X_test.iterrows():
        if index in index_list:
            label_list.append('True')
        else:
            label_list.append('False')
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    count = 0
    # Calculating statistics for simple classification
    for index, value in enumerate(X_test['Class']):
        if value == 0 and label_list[index] == 'True':
            if label_list[index] == 'True':
                true_pos += 1
                count += 1
        elif value == 1 and label_list[index] == 'False':
                true_neg += 1
                count += 1
        elif value == 1 and label_list[index] == 'True':
                false_pos += 1
                count += 1
        elif value == 0 and label_list[index] == 'False':
                false_neg += 1
                count += 1
    """Question 2.4"""
    simple_acc = round(((true_pos + true_neg) / count), 2)
    simple_tpr = round((true_pos / (true_pos+false_neg)), 2)
    simple_tnr = round((true_neg / (true_neg + false_pos)), 2)


    """Question 2.5"""
    print("--- Question 2.5 ---")
    print("\n")
    stats_list = [true_pos, false_pos, true_neg, false_neg, simple_acc,
                  simple_tpr, simple_tnr]
    table_printer(stats_list)
    print("\n")

    """Question 2.6"""
    print("---Question 2.6 ---")
    print("\nMy Classifier is better than a coin flip with a 70% accuracy.")

