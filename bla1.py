

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
import csv_handle as csv_org
#
# def filter_data_by_feature(df,colName,feat_name):
#     # from all data get data only on spesifc country
#
#     # df.sort_values(by=col_name, ascending=False)
#     by_country = df.loc[df[colName].isin(feat_name)]
#     return by_country


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


# Function to perform training with giniIndex.
def train_using_gini(X_train, X_test, y_train):
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion="gini",
                                      random_state=100, max_depth=3, min_samples_leaf=5)

    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini


# Function to make predictions
def prediction(X_test, clf_object,y_test):
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print("y_pred:",y_pred,len(y_pred))
    print("y_test:", y_test, len(y_test))
    return y_pred


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
    if not isinstance(y_test, bytearray):
        y_test = np.asarray(y_test)
        # print(y_test, len(y_test))
    #classification, the count of true negatives is Matrix[0,0] , false negatives is Matrix[1,0] , true positives is Matrix[1,1]  and false positives is Matrix[0,1] .
    matrix=confusion_matrix( y_test, y_pred)
    print("Confusion Matrix:",matrix)
    # TN=matrix[0,0]
    # FN=matrix[1,0]
    # FP = matrix[0, 1]
    # TP = matrix[1, 1]
#computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
    print("Accuracy : ",
          accuracy_score(y_test, y_pred) * 100)


    """
    The reported averages include:
    precision    recall  f1-score   support
    micro average -averaging the total true positives, false negatives and false positives
    macro average-averaging the unweighted mean per label,
    weighted average -averaging the support-weighted mean per label
    and sample average -only for multilabel classification- NOT IN  OUR CASE
    """
    print("Report :",
          classification_report(y_test, y_pred))






################################



#--------------------------------------
def accuracy(right,all_test):
    a=right/all_test
    return a

#-----------------------------------
def error(accuracy):
    err=1-accuracy
    return err

#----------------------------------
def recall(TP,FN):
    rec=TP/(TP+FN)
    return rec

#----------------------------------
def precision(TP,FP):
    if TP+FP ==0:# Prevents zero division
        return np.nan
    pre=TP/(TP+FP)
    return pre

#----------------------------------
def F_score(recall,precision):
    if recall == 0 or precision==0:# Prevents zero division
        return np.nan
    rec =1/recall
    pre=1/precision
    if rec+pre == 0:# Prevents zero division
        return np.nan
    f_score=2/(rec+pre)
    return f_score

#--------------------------------
# true positive rate-recall
def TPR(TP,FN):
    tpr=TP/(TP+FN)
    return tpr

#--------------------------------
## false positive rate
def FPR(FP,TN):
    fpr=FP/(FP+TN)
    return fpr





###########################
def main():
    data = importdata()
    fillter_col = 'native-country'
    fillter_feat = [' Cuba']

    data = csv_org.filter_data_by_feature(data, fillter_col, fillter_feat)
    col_to_split = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'result']
    for col_name in col_to_split:  # run on the num of cols
        col = LabelEncoder()
        data[col_name + '_n'] = col.fit_transform(data[col_name])
        data[col_name + '_n'] = col.fit_transform(data[col_name])
    data1 = data.drop(col_to_split, axis='columns')
    data1 = data1.drop('native-country', axis='columns')
    print(data1)
    data1.to_csv('1.csv', index=False)

    X, Y, X_train, X_test, y_train, y_test = splitdataset(data1)
    print(X_test)
    clf_gini = train_using_gini(X_train, X_test, y_train)

    print("Results Using Gini Index:")
    print(len(X_test))
    y_pred_gini = prediction(X_test, clf_gini,y_test)
    cal_accuracy(y_test, y_pred_gini)






# Calling main function
if __name__ == "__main__":
    main()
    print("")
