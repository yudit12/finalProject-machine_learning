import numpy as np
import pandas as pd
import csv_handle as csv_org
import lgReg_handle as lgr

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
#enter needed val in the  new cols for ccls that opens for more the 2 options
def enter_val_toCol(df, name,feat):
    length = df.shape[0]
    # print(feat)
    col_val = list(df[name])
    # print(feat)
    index = df.index.tolist()
    for i in range(length):
        for v in feat:
            if col_val[i] == v:  # feat name = val in the col list
                if v != ' ?':
                    df[v].at[index[i]] = 1

            else:  # feat name != val in the col list
                if v != ' ?':
                    df[v].at[index[i]] = 0
                    if col_val[i] == ' ?':  # if ther is '?'  in thlist col enter in the split col -1
                        df[v].at[index[i]] = -1


#--------------------------------------------------
def handle_2optionCol(df,name,feat):
    index = df.columns.get_loc(name)  # index  in the real df
    # df.insert(index , name + '1', np.nan)
    col_val = list(df[name])
    index = df.index.tolist()
    length = df.shape[0]
    for i in range(length):
        if feat[0] == col_val[i]:
            df[name ].at[index[i]] = 1
        else:
            df[name ].at[index[i]] = 0


#-----------------------------------------------
# insert on col to df split to the col feathers
def insert_col(df,name,start_col,feat):
    # add new cols

    for val in feat:
        if val!=' ?':
            df.insert(start_col, val,np.nan)
            start_col = start_col + 1
    #enter needed val in the  new cols
    enter_val_toCol(df, name, feat)


    # df.to_csv('data.csv', index=False)




#-----------------------------------------------------------
def del_col(df,col_list):
    for val in col_list:
        feat = get_col_feat(df, val)
        if len(feat) > 2:
            df.__delitem__(val)
    df.__delitem__('native-country')


#---------------------------------------------------------
# get list of col to split and  insert them to the df
def insert_all_col(df,col_list):
    for val  in col_list:
        feat=get_col_feat(df,val)
        # print(feat, len(feat))
        # print(feat)
        if len(feat)>2:
            # index, name = get_col_name(df, val)  # index= num of col in real loc of df
            index = df.columns.get_loc(val)
            insert_col(df,val, index + 1, feat)
        if len(feat)==2:
            handle_2optionCol(df, val, feat)




    del_col(df, col_list)
   # df.to_csv('data.csv', index=False)








    return df

#----------------------------------------------------------------
def main():
    path = 'economic_data.csv'
    df = pd.read_csv(path)
    country_name = [' Cuba']
    df=sort_data_by_country(df, country_name)
    print(df.shape)
    col_to_split = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex','native-country', 'result']
    insert_all_col(df, col_to_split)
    # print(df.shape)
    # col_num = df.columns.get_loc('workclass')
    # print(col_num)
    #
    print(df.shape)
    csv_org.two_option_col('result',' <=50K',' >50K', df)
    csv_org.two_option_col('sex',' Male',' Female', df)

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

