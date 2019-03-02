from sklearn.preprocessing import LabelEncoder
import pandas as pd
import csv_handle as csv_org
from sklearn import tree
import sklearn.datasets as datasets
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
import numpy as np

df = pd.read_csv('economic_data.csv')
#df.head()

fillter_col = 'native-country'
fillter_feat = [' Cuba']
df=csv_org.filter_data_by_feature(df, fillter_col,fillter_feat)


#age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,result
col_to_split = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'result']

df = df.drop('native-country',axis='columns')
for col_name in col_to_split:  # run on the num of cols
    col=LabelEncoder()
    df[col_name+'_n'] = col.fit_transform(df[col_name])
print(df)
print(col_to_split)
df_n = df.drop(col_to_split,axis='columns')

XMatrix = csv_org.x_matrix(df_n)
print("XMatrix")
print(XMatrix)
y = csv_org.y_vector(df_n)

model = tree.DecisionTreeClassifier()
model.fit(XMatrix, y)
score=model.score(XMatrix, y)#score check how match the model appropriate to the test data
print(score)
print(XMatrix[1])
predict=model.predict([XMatrix[1]])#predict ask if the data of one row his result is 0/1
print('predict',predict)
print(df_n)

df_n.to_csv('dt3.csv', index=False)

"""
n_nodes = model.tree_.node_count
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
children_left = model.tree_.children_left
children_right = model.tree_.children_right
feature = model.tree_.feature
threshold = model.tree_.threshold
stack = [(0, -1)]  # seed is the root node id and its parent depth
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    # If we have a test node
    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True

for i in range(n_nodes):
    if is_leaves[i]:
        print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
    else:
        print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
              "node %s."
              % (node_depth[i] * "\t",
                 i,
                 children_left[i],
                 feature[i],
                 threshold[i],
                 children_right[i],
                 ))
"""
dot_data = StringIO()
print(dot_data)
export_graphviz(model, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#Image(graph.create_png())