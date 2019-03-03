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
    path = 'countries_tree'
    os.path.dirname(path)
    if not os.path.exists(path):
        os.makedirs(path)

    os.chdir(path)
#---------------------------------------
def splitData(file, country_name, col_to_split):
    XMatrix, y, data_feature_names = orderDataForCountry(file, country_name, col_to_split)

    X_train, X_test, y_train, y_test = train_test_split(XMatrix, y, test_size=0.3, random_state=100)
    return X_train, X_test, y_train, y_test,data_feature_names

def runAllCountries(file,col_to_split,country_compare_modle):#fillter_col = 'native-country'

    countries=list(set(list(file['native-country'])))

    # del  the native-country  col from  the cuntry list
    if('native-country' in col_to_split):
        col_to_split.remove('native-country');

    compare_modle=np.nan
    compareX_test=[]
    compareY_test = []
    for country_name in countries:#all countries without USA and '?'

        if country_name==' United-States' or country_name==' ?' or country_name==' Holand-Netherlands':
            continue
        else:
            print('---------------------------------------------------')
            print('---------country_name', country_name,'---------')
            X_train, X_test, y_train, y_test,data_feature_names=splitData(file, country_name, col_to_split)

            model,typeModel=treeForCountry(country_name,X_train, y_train,data_feature_names)

            if country_name==country_compare_modle[0]:
                compare_modle=model
                compareX_test=X_test
                compareY_test = y_test


            if error.calc_error(X_test, model, y_test,flag=0)==-1:
                continue

    print('#############################################\n'
          '#########     compare models    #############\n'
          '#############################################')
    if  compare_modle!=np.nan and compareX_test!=[] and compareY_test!=[]:
        errorTree(country_compare_modle,typeModel, compare_modle, compareX_test, compareY_test)


#--------------------------------------------------


def orderDataForCountry(file,country_name,col_to_split):
    fillter_col = 'native-country'
    fillter_feat = [country_name]
    df = csv_org.filter_data_by_feature(file, fillter_col, fillter_feat)
    df = df.drop('native-country', axis='columns')

    for col_name in col_to_split:  # run on the num of cols
        col = LabelEncoder()
        df[col_name + '_n'] = col.fit_transform(df[col_name])

    df_n = df.drop(col_to_split, axis='columns')
    XMatrix = csv_org.x_matrix(df_n)

    y = csv_org.y_vector(df_n)
    data_feature_names = list(df_n)[:-1]


    return XMatrix,y,data_feature_names


#--------------------------------------------------
def treeForCountry(country_name, X_train,y_train,data_feature_names):
        typeModel="gini"#gini/entropy
        clf = tree.DecisionTreeClassifier(criterion=typeModel, random_state=100)

        clf.fit(X_train,y_train)

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
        return clf,typeModel





def errorTree(country_name,typeModel, clf,X_test, y_test):
    accr, rec, pre, f_sc, tpr, fpr=error.calc_error(X_test, clf, y_test,flag=1)
    print('--------DecisionTree type',typeModel,'---------------')
    modelName=' DecisionTreeClassifier with '
    error.printResult(country_name[0],modelName , accr, rec, pre, f_sc, tpr, fpr)


