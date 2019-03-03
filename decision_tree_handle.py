import pandas as pd
import csv_handle as csv_org
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pydotplus
from sklearn import tree
import collections
import os
import error_handle as error
import numpy as np

def changePath():
    path = 'countries_tree_gini'
    os.path.dirname(path)
    if not os.path.exists(path):
        os.makedirs(path)

    os.chdir(path)
#---------------------------------------
def runAllCountries(file,col_to_split):#fillter_col = 'native-country'

    # countries=list(set(list(file['native-country'])))
    countries = [' Cuba']
    # del  the native-country  col from  the cuntry list
    if('native-country' in col_to_split):
        col_to_split.remove('native-country');



    for country_name in countries:#all countries without USA and '?'
        if country_name==' United-States' or country_name==' ?' or country_name==' Holand-Netherlands':
            continue
        else:
            XMatrix, y,data_feature_names= orderDataForCountry(file,country_name,col_to_split)

            X_train, X_test, y_train, y_test = train_test_split(XMatrix, y, test_size=0.3, random_state=100)
            print(X_test,len(X_test))
            # print(country_name)
            # print(XMatrix)
            model=treeForCountry(country_name,X_train, y_train,data_feature_names)
            y_pred=error.prediction(X_test, model, y_test)
            error.cal_accuracy(y_test, y_pred,X_test)


#--------------------------------------------------



####

def orderDataForCountry(file,country_name,col_to_split):
    fillter_col = 'native-country'
    fillter_feat = [country_name]
    df = csv_org.filter_data_by_feature(file, fillter_col, fillter_feat)
    #col_to_split = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'result']
    df = df.drop('native-country', axis='columns')


    #print('df')
    #print(df)
    for col_name in col_to_split:  # run on the num of cols
        col = LabelEncoder()
        df[col_name + '_n'] = col.fit_transform(df[col_name])

    # print(col_to_split)
    df_n = df.drop(col_to_split, axis='columns')
    XMatrix = csv_org.x_matrix(df_n)

    y = csv_org.y_vector(df_n)
    data_feature_names = list(df_n)[:-1]

#    df_n.to_csv('dt3.csv', index=False)
#     print(df)
    return XMatrix,y,data_feature_names

# Function to perform training with giniIndex.
def train_using_gini(X_train,  y_train):
    # Creating the classifier object
    clf_gini = tree.DecisionTreeClassifier(criterion="gini")
    # Performing training
    clf_gini.fit(X_train,y_train)
    return clf_gini

#--------------------------------------------------
def treeForCountry(country_name, X_train,y_train,data_feature_names):

        clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=100)

        clf = clf.fit(X_train,y_train)
        # clf=train_using_gini(X_train, y_train)

        # Visualize data
        dot_data = tree.export_graphviz(clf,
                                        feature_names=data_feature_names,
                                        out_file=None,
                                        filled=True,
                                        rounded=True)
        graph = pydotplus.graph_from_dot_data(dot_data)

        colors = ('turquoise', 'orange')
        edges = collections.defaultdict(list)

        for edge in graph.get_edge_list():
            edges[edge.get_source()].append(int(edge.get_destination()))

        for edge in edges:
            edges[edge].sort()
            for i in range(2):
                dest = graph.get_node(str(edges[edge][i]))[0]
                dest.set_fillcolor(colors[i])

        name_img='tree%s.png'%country_name
        graph.write_png(name_img)
        return clf



