from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import csv_handle as csv_org
import error_handle as error
# Function importing Dataset
def importdata():
    path = 'economic_data.csv'
    balance_data = pd.read_csv(path)
    # Printing the dataswet shape
    print("Dataset Lenght: ", len(balance_data))
    print("Dataset Shape: ", balance_data.shape)
    # print(balance_data)
    return balance_data


# Function to split the dataset
def splitdataset(balance_data):
    # Seperating the target variable
    X = balance_data.values[:, 0:13]
    # print(X)
    Y = balance_data.values[:, 13]
    # print(Y)

    # Spliting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=100)

    return X, Y, X_train, X_test, y_train, y_test


def main():
    data = importdata()
    fillter_col = 'native-country'
    fillter_feat = [' Cuba']

##########
    # #change data string to labels
    # data = csv_org.filter_data_by_feature(data, fillter_col, fillter_feat)
    # col_to_split = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'result']
    # for col_name in col_to_split:  # run on the num of cols
    #     col = LabelEncoder()
    #     data[col_name + '_n'] = col.fit_transform(data[col_name])
    #     data[col_name + '_n'] = col.fit_transform(data[col_name])
    # data1 = data.drop(col_to_split, axis='columns')
    # data1 = data1.drop('native-country', axis='columns')
    # # print(data1)
    # data1.to_csv('1.csv', index=False)
##########
    path = 'economic_data.csv'
    df_org = pd.read_csv(path)

    fillter = [' Cuba']
    df = csv_org.filter_data_by_feature(df_org, 'native-country', fillter)
    df = df.reset_index(drop=True)
    df.to_csv('data.csv', index=False)

    col_to_split = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                    'native-country', 'result']
    csv_org.insert_all_col(df, col_to_split)

    csv_org.normalizationAll(df)  # normalization data


#####################3
    XMatrix = csv_org.x_matrix(df)
    y = csv_org.y_vector(df)
    X, Y, X_train, X_test, y_train, y_test = splitdataset(df)
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
    clf.fit(X, y)
    # print(X_train)
    # logreg =LogisticRegression(C=0.5, solver='lbfgs', penalty='l2').fit(XMatrix,y)
    # print(logreg)
    # y_pred=error.prediction(XMatrix, logreg,y )
    # error.cal_accuracy(y_test, y_pred, X_test)

# Calling main function
if __name__ == "__main__":
    main()


