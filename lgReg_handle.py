from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np


def divDataByKFold(XMatrix, y, k_parameter):  # div data to test and train by k fold

    kf = KFold(n_splits=k_parameter)  # Define the split - into 10 folds
    kf.get_n_splits(XMatrix)  # returns the number of splitting iterations in the cross-validator

    X_train_matrix = []
    X_test_matrix = []
    y_train_matrix = []
    y_test_matrix = []

    for train_index, test_index in kf.split(XMatrix):
        # Each iteration get new: X_train,y_train,X_test,y_test

        X_train = []
        X_test = []
        y_train = []
        y_test = []
        for i in range(train_index.__len__()):
            indexTrain = train_index[i]
            X_train.append(XMatrix[indexTrain])
            y_train.append(y[indexTrain])

        for j in range(test_index.__len__()):
            indexTest = test_index[j]
            X_test.append(XMatrix[indexTest])
            y_test.append(y[indexTest])

        X_train_matrix.append(X_train)
        X_test_matrix.append(X_test)
        y_train_matrix.append(y_train)
        y_test_matrix.append(y_test)

    return (X_train_matrix, X_test_matrix, y_train_matrix, y_test_matrix)


# ------------------------------------------------------------------------
def average(vec):  # average of vector
    sum = 0
    length = len(vec)
    for i in range(length):
        sum += vec[i]
    average = sum / length
    return average


def indexMinElement(vec):  # return the index of min element in vector
    minVal = vec[0]
    length = len(vec)
    indexMin = 0
    for i in range(1, length):
        if minVal > vec[i]:
            minVal = vec[i]
            indexMin = i

    return indexMin


# -------------------------------------
def k_fold_cross_validation(X_train_matrix, y_train_matrix, X_test_matrix, y_test_matrix, k_parameter=10):
    # C_param_range = [ np.inf, 10, 1, 0.5,(1/3),0.1]
    C_param_range = [1, 0.5, (1 / 3), 0.1]
    testErrOneModel = [0.0] * k_parameter
    testErrAllModels = []

    for c in C_param_range:
        for i in range(k_parameter):
            logreg = LogisticRegression(C=c, solver='lbfgs', penalty='l2').fit(X_train_matrix[i], y_train_matrix[i])
            errI = logreg.predict(X_test_matrix[i])
            testErrOneModel[i] = float(sum(errI != y_test_matrix[i])) / len(y_test_matrix[i])
        avgErr = np.mean(testErrOneModel)
        print('The average error value of lambda=', 1 / c, 'is', avgErr)
        testErrAllModels.append(avgErr)

    indexBetterModel = indexMinElement(testErrAllModels)

    optimalLambda = C_param_range[indexBetterModel]
    print('')
    print('The optimal lambda', 1 / optimalLambda)
    print('The average error of the model with this lambda is:', testErrAllModels[indexBetterModel])
    #  print('The average error of the model with lambda=0 is:', testErrAllModels[0])

    return (C_param_range, testErrAllModels, optimalLambda)


# ------------------------------------------------------------------------
def draw_graph(C_param_range, testErrAllModels):
    C_param_range = C_param_range[1:]
    c_len = len(C_param_range)
    lambdaArray = []
    for i in range(c_len):
        lambdaArray.append(1 / C_param_range[i])

    testErrAllModels = testErrAllModels[1:]

    plt.title(" error for given lambdas")
    plt.plot(lambdaArray, testErrAllModels, label='test error')
    plt.xlabel('lambdas')
    plt.ylabel('erors')

    plt.legend()
    plt.show()


def graph_learning_groups(XMatrix, y, optimalLambda, numAllRow):
    learNum = int(numAllRow * 0.7)
    learNum = int(learNum / 10) * 10 + 1
    print("learNum", learNum)
    print("rang ", list(range(10, learNum, 10)))
    learningGroups = np.array(range(10, learNum, 10))
    errAvg = []
    errAvgTrain = []
    xTestMatrix = XMatrix[learNum:]
    yTestVec = y[learNum:]
    print("len test", len(yTestVec))
    for rowTrains in learningGroups:
        xTrainMatrix = XMatrix[0:rowTrains]
        yTraintVec = y[0:rowTrains]
        logreg = LogisticRegression(C=optimalLambda, solver='lbfgs', penalty='l2').fit(xTrainMatrix, yTraintVec)
        errTest = logreg.predict(xTestMatrix)
        errAvg.append(float(sum(errTest != yTestVec)) / len(yTestVec))

        errTrain = logreg.predict(xTrainMatrix)
        errAvgTrain.append(float(sum(errTrain != yTraintVec)) / len(yTraintVec))

    plt.plot(learningGroups, errAvg, label='test error')
    plt.plot(learningGroups, errAvgTrain, label='train error')
    plt.xlabel('number of example')
    plt.ylabel('errors')

    plt.legend()
    plt.show()


