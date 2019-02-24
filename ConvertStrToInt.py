import numpy as np

#---------------------------------------
#A method that checks whether the cell is empty- no information
def isNaN(val):
    return val != val or val==''
#---------------------------------------

#Returns col content
def contain_col(col_num,df):
    colList = []
    for j in range(df.shape[0]):# run on the num of rows
        #col+=str(df.iloc[j][col_num])+" \n"
        colList.append(df.iloc[j][col_num])
    return colList
#------------------------------------------------

#A method that splits the column with string categories  the colums as number of categories
# it has, each new column receives the value of the category,
# the cells in the column are marked with one where the category is the same in the original column and the rest  is zero
def split_col_data(col_name, df):
    length = df.shape[0]
    col_num = df.columns.get_loc(col_name)
    index = df.index.tolist()
    col = contain_col(col_num, df)

    if col_name == 'result':
        indexCol = 0
        for i in index:
            if col[indexCol] == ' <=50K':  # Less than 50k or equal
                df['result'].at[i] = 0
            elif col[indexCol] == ' >50K':
                df['result'].at[i] = 1  # more than 50k
            indexCol += 1


    #print(df)