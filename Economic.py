import numpy as np
import pandas as pd
import csv_handle as csv_org
import lgReg_handle as lgr
from sklearn import tree
from sklearn.datasets import load_iris
import graphviz
import decision_tree_handle as dtree
import tree2 as t2

def main():

    '''
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)

    clf.predict([[2., 2.]])
    #clf.predict_proba([[2., 2.]])
    print(clf)
    #--------------------
    '''
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


    XMatrix = csv_org.x_matrix(df)
    y = csv_org.y_vector(df)
    # print(df)

    X_train_matrix, X_test_matrix, y_train_matrix, y_test_matrix \
        = lgr.divDataByKFold(XMatrix, y, k_parameter=10)  # Define the split - into 10 folds

    C_param_range, testErrAllModels,optimalLambda=\
        lgr.k_fold_cross_validation(X_train_matrix, y_train_matrix, X_test_matrix,y_test_matrix, k_parameter=10)

    lgr.draw_graph(C_param_range, testErrAllModels)
    lgr.graph_learning_groups(XMatrix,y,optimalLambda,len(y))

    # dtree.changePath()
    # dtree.runAllCountries(df_org,col_to_split)

if __name__ == "__main__":
    main()


    '''   fillter_col='native-country'
   fillter_feat = [' Cuba']

   df=csv_org.filter_data_by_feature(df_org, fillter_col,fillter_feat)
   print(df_org)
   col_to_split = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex','native-country', 'result']
   csv_org.insert_all_col(df, col_to_split)
   #print(df)
   csv_org.normalizationAll(df)# normalization data
#    df.to_csv('data.csv', index=False)
   XMatrix = csv_org.x_matrix(df)
   y = csv_org.y_vector(df)

   print(XMatrix)
'''


