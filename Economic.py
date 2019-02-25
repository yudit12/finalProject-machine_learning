import numpy as np
import pandas as pd
import csv_handle as csv_org
import lgReg_handle as lgr


def main():
    path = 'economic_data.csv'
    df = pd.read_csv(path)
    country_name = [' Cuba']
    df=csv_org.sort_data_by_country(df, country_name)
    # print(df.shape)
    col_to_split = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex','native-country', 'result']
    csv_org.insert_all_col(df, col_to_split)
    csv_org.normalizationAll(df)# normalization data

    # print(df.shape)
    # col_num = df.columns.get_loc('workclass')
    # print(col_num)
    #


    df.to_csv('data.csv', index=False)


    XMatrix = csv_org.x_matrix(df)
    y = csv_org.y_vector(df)

    X_train_matrix, X_test_matrix, y_train_matrix, y_test_matrix \
        = lgr.divDataByKFold(XMatrix, y, k_parameter=10)  # Define the split - into 10 folds





#----------------------------------------------------------------------------------------
    # csv_org.split_col_data('result', df)
    #
    #
    #
    # XMatrix = csv_org.x_matrix(df)
    # y = csv_org.y_vector(df)
    #
    # X_train_matrix, X_test_matrix, y_train_matrix, y_test_matrix \
    #     = lgr.divDataByKFold(XMatrix, y, k_parameter=10)  # Define the split - into 10 folds
    #
    #
    #
    # ''' C_param_range, testErrAllModels,optimalLambda=\
    #     lgr.k_fold_cross_validation(X_train_matrix, y_train_matrix, X_test_matrix,y_test_matrix, k_parameter=10)
    #
    #
    # lgr.draw_graph(C_param_range, testErrAllModels)
    # lgr.raph_learning_groups(XMatrix,y,optimalLambda)'''


if __name__ == "__main__":
    main()

