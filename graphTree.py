'''
import sklearn.datasets as datasets
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

iris=datasets.load_iris()
df=pd.DataFrame(iris.data, columns=iris.feature_names)
y=iris.target
dtree=DecisionTreeClassifier()
dtree.fit(df,y)

dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
print("4")
'''


import pandas as pd
import csv_handle as csv_org
from sklearn.preprocessing import LabelEncoder

import pydotplus
from sklearn.datasets import load_iris
from sklearn import tree
import collections

import cv2
import os
import urllib

df_org = pd.read_csv('economic_data.csv')
#df.head()

path = 'countries_tree_gini'
os.chdir(path)

fillter_col = 'native-country'
countries=list(set(list(df_org['native-country'])))
#countries=  [' Greece']
print("len",len(countries))
#countriesMin=[]#all countries without USA and '?'
for country_name in countries:#all countries without USA and '?'
    if country_name==' United-States' or country_name==' ?':
        continue
    else:
        fillter_feat = [country_name]
        df = csv_org.filter_data_by_feature(df_org, fillter_col, fillter_feat)
        #countriesMin.append(country_name)

        #fillter_feat = [' Greece']
        #df=csv_org.filter_data_by_feature(df, fillter_col,fillter_feat)


        #age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,result
        col_to_split = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'result']

        df = df.drop('native-country',axis='columns')

        #data_feature_names=list(df)[:-1]
        #print("name:" ,data_feature_names)
        for col_name in col_to_split:  # run on the num of cols
            col=LabelEncoder()
            df[col_name+'_n'] = col.fit_transform(df[col_name])

        #print(col_to_split)
        df_n = df.drop(col_to_split,axis='columns')
        print(df)
        XMatrix = csv_org.x_matrix(df_n)
        print("XMatrix")
        print(XMatrix)
        y = csv_org.y_vector(df_n)
        data_feature_names=list(df_n)[:-1]

        df_n.to_csv('dt3.csv', index=False)

        # Training
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(XMatrix,y)

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

        #img = cv2.imread(name_img, 1)

        name_img='tree%s.png'%country_name
        graph.write_png(name_img)


        #cv2.imwrite(os.path.join(path, 'tree%s.png'%country_name), graph)
        #cv2.waitKey(0)
        #urllib.urlretrieve('url of picture' 'name of file to save to')


