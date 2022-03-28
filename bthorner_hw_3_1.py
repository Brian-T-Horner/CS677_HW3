"""
Brian Horner
CS 677 - Summer 2
Date: 7/27/2021
Week 3 Homework Question 1
This program loads in the Bank Note data and finds the mean and standard
deviation of each feature for the valid bank notes, fake Bank notes and all
bank notes.
"""

# Imports
import pandas as pd
import numpy as np
import statistics as st




def mean_calc(dataframe, column, classvalue='All'):
    """Calculates the mean of a column of a dataframe. Has the option limit
    based on the class value with passed in classvalue argument."""
    if classvalue == 'All':
         return round(st.mean(dataframe[column].values), 2)
    else:
        current_df = dataframe[dataframe['Class'] == classvalue]
        return round(st.mean(current_df[column].values), 2)


def standard_deviation(dataframe, column, classvalue='All'):
    """Calculates the standard deviation of a column of a dataframe. Has the
    option limit based on the class value with passed in classvalue argument."""
    if classvalue == 'All':
        return round(st.stdev(dataframe[column].values), 2)
    else:
        current_df = dataframe[dataframe['Class'] == classvalue]

        return round(st.stdev((current_df[column].values)), 2)



def table_printer():
    """Formats the computaations for table printing."""
    header_list = ['class', 'u(f1)', 'o(f1)', 'u(f2)', 'o(f2)', 'u(f3)', 'o(f3)',
               'u(f4)','o(f4)']
    column_list = ['Variance', 'Skewness', 'Curtosis', 'Entropy']
    true_list = ['0', ]
    false_list = ['1']
    all_list = ['All']
    print_list = []
    print("--- Mean and Standard Deviations of Features for Genuine, Forged "
          "and All Bank Notes ---")
    for item in column_list:
        true_list.append(mean_calc(df, item, 0))
        true_list.append(standard_deviation(df, item, 0))
        false_list.append(mean_calc(df, item, 1))
        false_list.append(standard_deviation(df, item, 1))
        all_list.append(mean_calc(df, item))
        all_list.append(standard_deviation(df, item))
    print_list.append(true_list)
    print_list.append(false_list)
    print_list.append(all_list)
    print_list.insert(0, list(header_list))



    print(f"---Breakdown of True and Fake Bank Notes Features Mean and "
          f"Standard Deviations.")
    print("Features are as follows: 'Variance', 'Skewness', 'Curtosis', "
          "'Entropy'.")
        # Enumerating over print list
    for index, stuff in enumerate(print_list):
        # Adding a | in front of each value of the lists in print list
        row = '|'.join(str(value).ljust(12) for value in stuff)
        # Printing the row for the list in print list
        print(row)
        # Adding a line between the header and the data rows
        if index == 0:
            print('-' * len(row))


if __name__ == "__main__":

    formatter = '--------------------------------------------------------------'
    """Question 1.1"""
    # Reading data as a csv and adding columns
    df = pd.read_csv('data_banknote_authentication.txt', header=None)
    df.columns = ['Variance', 'Skewness', 'Curtosis', 'Entropy', 'Class']

    # Adding 'Color' column
    df['Color'] = np.where(df['Class'] == 0, 'Green', 'Red')

    # Methods to make sure data looks how we expect
    # print(df.head())
    # print(df.shape)

    # Making sure counts for real and fake notes are correct
    truenote = df[df['Class'] == 0]
    fakenote = df[df['Class'] == 1]
    # print(f"True note shape {truenote.shape}.")
    # print(f"Fake note shape{fakenote.shape}.")

    """Question 1.2"""
    print('\t')
    print('--- Question 1.2 ---')
    table_printer()
    print(formatter)

    """Question 1.3"""
    print('\t')
    print('--- Question 1.3 ---')
    print(formatter + "\n" + "Question 1.3 Below:")
    print(f"There is a high standard deviation for Skewness and "
    f"Curtosis than Variance and Entropy. The lowest standard deviation for the" 
    f"Real and Fake Bank Notes is Variance. ")




