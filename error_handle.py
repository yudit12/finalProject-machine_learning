
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
import csv_handle as csv_org

# Function to make predictions
def prediction(X_test, clf_object,y_test):
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print("y_pred:",y_pred,len(y_pred))
    print("y_test:", y_test, len(y_test))
    return y_pred


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred,x_test):

    y_test = np.asarray(y_test)

    #classification, the count of true negatives is Matrix[0,0] , false negatives is Matrix[1,0] , true positives is Matrix[1,1]  and false positives is Matrix[0,1] .

    matrix=confusion_matrix( y_test, y_pred)
    print("Confusion Matrix:",matrix)

#computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
    accuracy= accuracy_score(y_test, y_pred) * 100
    print("Accuracy : ",accuracy)
    print("error :",100-accuracy)
    my_error_calc(matrix, x_test,y_test)



    """
    The reported averages include:
    precision    recall  f1-score   support
    micro average -averaging the total true positives, false negatives and false positives
    macro average-averaging the unweighted mean per label,
    weighted average -averaging the support-weighted mean per label
    and sample average -only for multilabel classification- NOT IN  OUR CASE
    """
    print("Report :",
          classification_report(y_test, y_pred,target_names="1"))


#-------------------------------------

# my checkss
def my_error_calc(matrix,x_test,y_test):
 TN = matrix[0, 0]
 FN = matrix[1, 0]
 FP = matrix[0, 1]
 TP = matrix[1, 1]
 right = TP + TN
 wrong= FP+FN
 all_test = len(x_test)
 all_y=len(y_test)
 accr = accuracy(right, all_test)*100
 print("accr:", accr)


 print("error :", 100 - accr)
 rec = recall(TP, FN)
 print("recall :", rec)
 pre = precision(TP, FP)
 print("precision : ", pre)
 f_sc = F_score(rec, pre)
 print("F_score : ", f_sc)
 tpr = TPR(TP, FN)
 print("true positive rate-recall :", tpr)
 fpr = FPR(FP, TN)
 print("false positive rate :", fpr)

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

# --------------------------------