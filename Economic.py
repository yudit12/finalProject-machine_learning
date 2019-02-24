import numpy as np
import pandas as pd
import csv_handle as csv_org
#---------------------------------------------------------
#
def sort_data_by_country(df,country_name):
    # from all data get data only on spesifc country
    col_name = 'native-country'
    # df.sort_values(by=col_name, ascending=False)
    by_country = df.loc[df[col_name].isin(country_name)]
    return by_country
#-----------------------------------------------

def get_col_feat(df,col_n):
    feat_list = df[col_n].tolist()
    feat_list = list(set(feat_list))
    return feat_list

#---------------------------------------------------------

def main():
    path = 'economic_data.csv'
    df = pd.read_csv(path)
    country_name = [' Cuba']
    df=sort_data_by_country(df, country_name)
    print(df)
    addional_cols=get_col_feat(df, 'marital-status')
    print(addional_cols)


    XMatrix = csv_org.x_matrix(df)
    y = csv_org.y_vector(df)

    X_train_matrix, X_test_matrix, y_train_matrix, y_test_matrix \
        = lgr.divDataByKFold(XMatrix, y, k_parameter=10)  # Define the split - into 10 folds

if __name__ == "__main__":
    main()
