import numpy as np
import pandas as pd
import csv_handle as csv_org
import lgReg_handle as lgr
import ConvertStrToInt as strHandle

#---------------------------------------------------------
#
def sort_data_by_country(df,country_name):
    # from all data get data only on spesifc country
    col_name = 'native-country'
    # df.sort_values(by=col_name, ascending=False)
    by_country = df.loc[df[col_name].isin(country_name)]
    return by_country
#-----------------------------------------------
# returm the feat in the colum withot duplicates - to know how many column  add to the df
def get_col_feat(df,col_n):
    feat_list = df[col_n].tolist()
    feat_list = list(set(feat_list))
    feat_num= len(feat_list)
    return feat_list

#---------------------------------------------------------
# return col nums and threre indexses
def get_col_name(df,col_name):
    header = list(df)
    for val in header:
        index=header.index(val)
        name=val
        if col_name==name:
            return index,name
    return

#---------------------------------------------------------
# insert on col to df split to the col feathers
def insert_col(df,start_col,feat):
    for val in feat:
        if val!=' ?':
            df.insert(start_col, val, np.nan)
            start_col = start_col + 1
#    df.to_csv('data.csv',index=False)

#---------------------------------------------------------
# get list of col to split and  insert them to the df
def insert_all_col(df,col_list):
    for val  in col_list:
        feat=get_col_feat(df,val)
        index, name = get_col_name(df, val)  # index= num of col in real loc of df
        # print(feat, len(feat))
        insert_col(df, index + 1, feat)

#----------------------------------------------------------------
def main():
    path = 'economic_data.csv'
    df = pd.read_csv(path)
    country_name = [' Cuba']
    df=sort_data_by_country(df, country_name)
    # print(df)
    print(df.shape)

    col_to_split = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'result']
    insert_all_col(df, col_to_split)

    # print(df.shape)


    csv_org.split_col_data('result', df)



    XMatrix = csv_org.x_matrix(df)
    y = csv_org.y_vector(df)

    X_train_matrix, X_test_matrix, y_train_matrix, y_test_matrix \
        = lgr.divDataByKFold(XMatrix, y, k_parameter=10)  # Define the split - into 10 folds



    ''' C_param_range, testErrAllModels,optimalLambda=\
        lgr.k_fold_cross_validation(X_train_matrix, y_train_matrix, X_test_matrix,y_test_matrix, k_parameter=10)


    lgr.draw_graph(C_param_range, testErrAllModels)
    lgr.raph_learning_groups(XMatrix,y,optimalLambda)'''


if __name__ == "__main__":
    main()

