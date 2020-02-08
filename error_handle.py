
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import tree
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
     #print('TN , FN, FP,  TP ',TN , FN, FP,  TP)
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


def dif_alg_errors(logistic_error, entropy_error, gini_error):


    n_groups = 4

    # logistic = (82.8, 50, 60, 54.4)
    logistic= logistic_error

    std_logistic= (2, 3, 4, 1)

    # tree_entropy = (82.7, 95.65, 84.61, 89.7)
    tree_entropy=entropy_error
    std_entropy = (3, 5, 2, 3)
    tree_gini = gini_error
    std_gini = (2, 3, 4, 1)


    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.3

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = ax.bar(index, logistic, bar_width,
                    alpha=opacity, color='b',
                    yerr=std_logistic, error_kw=error_config,
                    label='logistic regression')

    rects2 = ax.bar(index + bar_width, tree_entropy, bar_width,
                    alpha=opacity, color='r',
                    yerr=std_entropy, error_kw=error_config,
                    label="tree-entropy")

    rects3 = ax.bar(index + bar_width*2, tree_gini, bar_width,
                    alpha=opacity, color='y',
                    yerr=std_gini, error_kw=error_config,
                    label="tree-gini")

    ax.set_xlabel('type')
    ax.set_ylabel('% error')
    ax.set_title('metrics error ')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(('accuracy', 'recall', 'precision', 'F_score'))
    ax.legend()

    fig.tight_layout()
    plt.show()


def graph_learning_groups(XMatrix,y,optimalLambda,numAllRow,model_name,type,max_depth):
    learNum=int(numAllRow*0.7)
    learNum=int(learNum/10)*10+1

    learningGroups=np.array(range(10,learNum,10))
    errAvg = []
    errAvgTrain=[]
    xTestMatrix=XMatrix[learNum:]
    yTestVec = y[learNum:]

    for rowTrains in learningGroups:
        xTrainMatrix = XMatrix[0:rowTrains]
        yTraintVec = y[0:rowTrains]
        if model_name== 'LogisticRegression':
            model = LogisticRegression(C=optimalLambda, solver='lbfgs', penalty='l2').fit(xTrainMatrix, yTraintVec)
        elif model_name=='DecisionTreeClassifier':
            model = tree.DecisionTreeClassifier(criterion=type, max_depth=max_depth,random_state=100).fit(xTrainMatrix, yTraintVec)
        errTest = model.predict(xTestMatrix)

        errAvg.append(float(sum(errTest != yTestVec)) / len(yTestVec))

        errTrain = model.predict(xTrainMatrix)

        errAvgTrain.append(float(sum(errTrain != yTraintVec)) / len(yTraintVec))

    plt.plot(learningGroups, errAvg, label='test error')
    plt.plot(learningGroups, errAvgTrain, label='train error')
    plt.xlabel('number of example')
    plt.ylabel('errors')

    plt.legend()
    plt.show()

# tree_type = 'entropy'
# logistic_error = (82.8, 50, 60, 54.4)
# entropy_error = (82.7, 95.65, 84.61, 89.7)
# gini_error = (75, 66.6, 44.4, 53.3)
# dif_alg_errors(logistic_error, entropy_error, gini_error)
# ------
# tree_type = 'gini'
# # logistic_error = (82.7, 33.3, 66.6, 44.4)

