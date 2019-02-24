
import csv
import numpy as np
import pandas as pd
from pandas import DataFrame



#----
yes=1
no=0
#----


#--------------------Methods-----------------------------
#A method that inserts columns into a table (for splitting the columns containing string categories)
# calls for a method that splits the relevant columns and deletes the original columns that was divided
def insert_col_df(df2):

    df2.insert(1, 'v', 1)
    df2.insert(5, 'typical', np.nan)
    df2.insert(6, 'asymptomatic', np.nan)
    df2.insert(7, 'nonanginal', np.nan)
    df2.insert(8, 'nontypical', np.nan)
    # --------------------------------------------
    df2.insert(19, 'fixed', np.nan)
    # df2.insert(19, 'NaN', np.nan)
    df2.insert(20, 'normal', np.nan)
    df2.insert(21, 'reversable', np.nan)
    # --------------------------------------------
    # df2.insert(23, 'Yes', np.nan)
    # df2.insert(24, 'No', np.nan)
    split_col_data('ChestPain', df2)
    split_col_data('Thal', df2)
    split_col_data('AHD', df2)
    # afther spliting the cols del the orginal
    df2.__delitem__('ChestPain')
    df2.__delitem__('Thal')
    df2.to_csv('data.csv', index=False)
    # print(df2.shape)
    # print(df2)

#-------------------------------------------------------------------
#Returns row content from the file (information about one patient)
def contain_row(row_num,df):
    rowList = []
    for j in range(df.shape[1]):# run on the num of cols
        if j==0:
            continue
        rowList.append(df.iloc[row_num][j])

    return rowList
#---------------------------------------
#Returns col content
def contain_col(col_num,df):
    colList = []
    for j in range(df.shape[0]):# run on the num of rows
        #col+=str(df.iloc[j][col_num])+" \n"
        colList.append(df.iloc[j][col_num])
    return colList

#---------------------------------------
#A method that checks whether the cell is empty- no information
def isNaN(val):
    return val != val or val==''
#-------------------------------


# method that does average on column values, does not consider empty cells
def averageCol(col):
    #didnt count nun lines, flagNaN is 1 if there are nan

    flagNaN=0;
    sum = 0
    line_count = 0
    for i in col:
        if not isNaN(i):
            sum += i
            line_count += 1
        else:
            flagNaN=1;

    if line_count == 0:
        return
    average = sum / line_count
    return (average,line_count,flagNaN)
#-------------------------------

#A method that receives a column number and returns the name of that column
def recognizeColByNum(file,col_num):

    header=list(file.columns.values)
    colName=header[col_num]

    return colName
#-------------------------------
#method that replaces an average nan value of the same column
def replaceNaN(file,col_num,avg):
    col = contain_col(col_num, file)  # create list from row
    rowNum=0
    colName= recognizeColByNum(file,col_num)

    for i in col:
        if isNaN(i):
            col = contain_col(col_num, file)
            file[colName].at[rowNum] = avg

        rowNum+=1

#-------------------------------

#A method that normalizes specific col in the file
def normalization(file, col_num):
    col = contain_col(col_num, file)  # create list from row
    average,line_count,flagNaN=averageCol(col)
    if flagNaN==1:
        replaceNaN(file,col_num, average)
        col = contain_col(col_num, file)  # create list from row
        average, line_count, flagNaN = averageCol(col)
    standard_deviation=0
    index=0
    normalization_col=col

    for i in col:
        tmp =i-average
        standard_deviation+= tmp**2#pow
    standard_deviation/=line_count;
    standard_deviation = standard_deviation**(0.5)


    for i in col:
        tmp= i - average
        tmp/= standard_deviation
        normalization_col[index]=tmp
        index+=1
    colName = recognizeColByNum(file, col_num)
    file[colName]=normalization_col#Updating the column to be normalized


# -------------------------------
#A method that splits the column with string categories  the colums as number of categories
# it has, each new column receives the value of the category,
# the cells in the column are marked with one where the category is the same in the original column and the rest  is zero
def split_col_data(col_name,df2):
     length= df2.shape[0]
     col_num =df2.columns.get_loc(col_name)
     index=df2.index.tolist()
     col = contain_col(col_num, df2)

     if col_name == 'result':
         indexCol=0
         for i in index:
             if col[indexCol] == ' <=50K':  # Less than 50k or equal
                 df2['result'].at[i] = 0
             elif col[indexCol] == ' >50K':
                 df2['result'].at[i] = 1  # more than 50k
             indexCol += 1



# -------------------------------


#A method that normalizes all cols in the file
def normalizationAll(file):
    colNum=file.shape[1]
    for colIndex in range(2,colNum-1):  # run on the num of cols WITHOUT V_ONE
        colLine = contain_col(colIndex, file)
        normalization(file, colIndex)

#---------------------------------
# method that return only x matrix from the ds (data frame) file
def x_matrix(file):

    X_mat=[]
    for i in range(file.shape[0]):  # run on the num of rows
        X_mat.append(xi_vector(file,i))
    return X_mat
#---------------------------------
#method that return the y vector - real answer
def y_vector(file):
    yVec=[]
    for i in range(file.shape[0]):  # run on the num of cols
        yVec.append(yi_val(file,i))
    return yVec
#---------------------------------
# method that return one row in the x-matrix (without the y v)
def xi_vector(file,i):
    X = contain_row(i, file)
    X = X[:-1]  # all row except the last cell
    return X
#---------------------------------
#method that return specific yi - spesific answer
def yi_val(file,i):
    yi = contain_row(i, file)
    yi = yi[-1]
    return yi