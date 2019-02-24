import numpy as np
import pandas as pd

#---------------------------------------------------------
def sort_data_by_country(df,country_name):
    # from all data get data only on spesifc country
    col_name = 'native-country'
    by_country = df.sort_values(by=col_name, ascending=False)
    by_country = df.loc[df[col_name].isin(country_name)]
    return by_country
#---------------------------------------------------------

def main():
    path = 'economic_data.csv'
    df = pd.read_csv(path)
    country_name = [' Cuba']
    df=sort_data_by_country(df, country_name)
    print(df)


    list1=[1, 2, 3]
    print(type(list1))
    num=3
    print(type(num))

    if isinstance(list1, list):
        print("y",list1[0])
    else:
        print("n")

    #listoflists = []
    sum=0
    listoflists= [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]

    for j in range(len(listoflists[0])):
        for i in range(len(listoflists)):
            sum+=listoflists[i][j]
        print(sum)
        sum=0
    # a_list = []
    # for i in range(0, 10):
    #     a_list.append(i)
    #     if len(a_list) > 3:
    #         a_list.remove(a_list[0])
    #         listoflists.append(list(a_list))
    print(listoflists)
    list1 = [1, 2, 3,1,3,5,7,2,2,2,7,"df",'df']
    print(list(set(list1)))

    #addingColumns(df)
    columns_of_string=["workclass","education","marital-status","occupation","relationship","race","native-country"]

    for column in range(len(columns_of_string)):
        col_vals=columns_of_string[column]
        num_different_val=len(list((set(col_vals))))
        print(num_different_val)



if __name__ == "__main__":
    main()