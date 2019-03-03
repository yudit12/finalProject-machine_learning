
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np

def calc_error(X_test, model, y_test,flag):
    #flag check if the method that call this function' is a mathod for comparing models
    y_pred = prediction(X_test, model, y_test)
    y_test, matrix, accuracy = cal_accuracy(y_test, y_pred)
    if matrix.shape == (1, 1):
        return -1;
    if (matrix[0][1] == 0 and matrix[1][1] == 0) or flag==1:
        accr, rec, pre, f_sc, tpr, fpr=my_error_calc(matrix, X_test,flag)
        return accr, rec, pre, f_sc, tpr, fpr
    elif (matrix[0][0] == 0 and matrix[1][0] == 0) or flag==1:
        accr, rec, pre, f_sc, tpr, fpr=my_error_calc(matrix, X_test,flag)
        return accr, rec, pre, f_sc, tpr, fpr
    else:
        understandable_method(y_test, y_pred)

def printResult(country_name,model_name,accr, rec, pre, f_sc, tpr, fpr):
    rec*=100
    pre*=100
    f_sc*=100
    tpr*=100
    fpr*=100

    print('Results of',model_name,'about the country',country_name)
    print('accuracy',accr,'\nrecall' ,rec,'\nprecision', pre, '\nF_score',f_sc,'\nTPR', tpr,'\nFPR', fpr)


# Function to make predictions
def prediction(X_test, clf_object,y_test):
    y_pred = clf_object.predict(X_test)
    return y_pred


def understandable_method(y_test, y_pred):
    target_names = ['class 0', 'class 1']
    y_test, matrix, accuracy = cal_accuracy(y_test, y_pred)
    report=classification_report(y_test, y_pred, target_names=target_names)
    print("Report :",report)


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):

    y_test = np.asarray(y_test)
    matrix=confusion_matrix( y_test, y_pred)
    #computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
    accuracy= accuracy_score(y_test, y_pred) * 100

    return y_test,matrix,accuracy
#-------------------------------------

# my checkss
def my_error_calc(matrix,x_test,flag):
     TN = matrix[0, 0]
     FN = matrix[1, 0]
     FP = matrix[0, 1]
     TP = matrix[1, 1]
     right = TP + TN
     all_test = len(x_test)
     accr = accuracy(right, all_test)*100
     rec = recall(TP, FN)
     pre = precision(TP, FP)
     f_sc = F_score(rec, pre)

     tpr = TPR(TP, FN)
     fpr = FPR(FP, TN)

     if flag==0:
         print("accr:", accr)
         if accr != np.nan:
             print("error :", 100 - accr)
         print("recall :", rec)
         print("precision : ", pre)
         print("F_score : ", f_sc)
         print("true positive rate-recall :", tpr)
         print("false positive rate :", fpr)


     return accr,rec,pre,f_sc,tpr,fpr
#--------------------------------------
def accuracy(right,all_test):
    if all_test==0:
        return np.nan
    a=right/all_test
    return a

#----------------------------------
def recall(TP,FN):
    if TP+FN==0:
        return np.nan
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
    if recall == np.nan or precision==np.nan:# Prevents zero division
        return np.nan
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
    if TP+FN==0:
        return np.nan
    tpr=TP/(TP+FN)
    return tpr

#--------------------------------
## false positive rate
def FPR(FP,TN):
    if FP+TN==0:
        return np.nan
    fpr=FP/(FP+TN)
    return fpr

# --------------------------------