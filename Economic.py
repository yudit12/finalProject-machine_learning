import numpy as np
import pandas as pd
import csv_handle as csv_org
import lgReg_handle as lgr
from sklearn import tree
from sklearn.datasets import load_iris
import graphviz
import decision_tree_handle as dtree

def main():
    path = 'economic_data.csv'
    df = pd.read_csv(path)

    '''  X = [[0, 0], [1, 1]]
    Y = [0, 1]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)

    clf.predict([[2., 2.]])
    #clf.predict_proba([[2., 2.]])
    print(clf)
    #--------------------

    iris = load_iris()
    clf_iris = tree.DecisionTreeClassifier()
    clf_iris = clf_iris.fit(iris.data, iris.target)
    print(clf_iris)
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("iris")
    '''

    dtree.build_tree();
    '''
    #fillter_feat = [' Cuba',' China',' Germany']
    # fillter_col='native-country'
    # fillter_feat = [' Germany']
    fillter_col='native-country'
    fillter_feat = [' Cuba']
    #fillter_col='sex'
    #fillter_feat = [' Female']
    df=csv_org.filter_data_by_feature(df, fillter_col,fillter_feat)

    # print(df.shape)
    df = df.reset_index(drop=True)
    df.to_csv('data.csv', index=False)
    #print(df)


    col_to_split = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex','native-country', 'result']
    csv_org.insert_all_col(df, col_to_split)
    csv_org.normalizationAll(df)# normalization data
    df.to_csv('data.csv', index=False)


    XMatrix = csv_org.x_matrix(df)
    y = csv_org.y_vector(df)

    X_train_matrix, X_test_matrix, y_train_matrix, y_test_matrix \
        = lgr.divDataByKFold(XMatrix, y, k_parameter=10)  # Define the split - into 10 folds

    C_param_range, testErrAllModels,optimalLambda=\
        lgr.k_fold_cross_validation(X_train_matrix, y_train_matrix, X_test_matrix,y_test_matrix, k_parameter=10)

    print("len ",len(y))
    lgr.draw_graph(C_param_range, testErrAllModels)
    lgr.graph_learning_groups(XMatrix,y,optimalLambda,len(y))
    '''

if __name__ == "__main__":
    main()

