import pandas as pd
import csv_handle as csv_org
import lgReg_handle as lgr
import decision_tree_handle as dtree
import error_handle as error

def main():

    path = 'economic_data.csv'
    df_org = pd.read_csv(path)

    country_name = [' Cuba',' Philippines',' Ecuador']

    df = csv_org.filter_data_by_feature(df_org, 'native-country', country_name)
    df = df.reset_index(drop=True)
   # df.to_csv('data.csv', index=False)

    col_to_split = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                    'native-country', 'result']
    csv_org.insert_all_col(df, col_to_split)
    # df.to_csv('data.csv', index=False)
    csv_org.normalizationAll(df)  # normalization data


    XMatrix = csv_org.x_matrix(df)
    y = csv_org.y_vector(df)

    print('#############################################\n'
          '########      LogisticRegression    #########\n'
          '#############################################')
    X_train, X_test, y_train, y_test \
        = lgr.divDataByKFold(XMatrix, y, k_parameter=10)  # Define the split - into 10 folds

    C_param_range, testErrAllModels,optimalLambda=\
        lgr.k_fold_cross_validation(X_train, y_train, X_test,y_test, k_parameter=10)

    lgr.draw_graph(C_param_range, testErrAllModels)
    lgr.graph_learning_groups(XMatrix,y,optimalLambda,len(y))




    print('#############################################\n'
          '####      DecisionTreeClassifier    #########\n'
          '#############################################')
    dtree.changePath()
    typeModel = "gini"  # gini/entropy
    #dtree.runAllCountries(df_org,col_to_split,country_name,typeModel)


    print('#############################################\n'
          '#########     compare models    #############\n'
          '#############################################')
    dtree.errorTree(country_name,typeModel,df_org,col_to_split)
    lgr.errorOfmodelOptimalLmbda(optimalLambda, XMatrix, y, country_name)
    tree_type = 'entropy'
    logistic_error = (82.8, 50, 60, 54.4)
    entropy_error = (82.7, 95.65, 84.61, 89.7)
    error.dif_alg_errors(logistic_error, entropy_error, tree_type)
    # ------
    tree_type = 'gini'
    logistic_error = (82.7, 33.3, 66.6, 44.4)
    gini_error = (75, 66.6, 44.4, 53.3,)
    error.dif_alg_errors(logistic_error, gini_error, tree_type)

if __name__ == "__main__":
    main()


