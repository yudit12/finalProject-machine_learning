
import csv
import numpy as np
import pandas as pd
from pandas import DataFrame



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
    two_option_col('ChestPain', df2)
    two_option_col('Thal', df2)
    two_option_col('AHD', df2)
    # afther spliting the cols del the orginal
    df2.__delitem__('ChestPain')
    df2.__delitem__('Thal')

    #df2.to_csv('data.csv', index=False)

#------------------------------------------------------------------
#A method that checks whether the cell is empty- no information
def isNaN(val):
    return val != val or val==''
#---------------------------------------


#---------------------------------------------------------

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
    col_val = list(df[name])
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
        if len(feat)>2:
            # index, name = get_col_name(df, val)  # index= num of col in real loc of df
            index = df.columns.get_loc(val)
            insert_col(df,val, index + 1, feat)
        if len(feat)==2:
            handle_2optionCol(df, val, feat)




    del_col(df, col_list)







    return df


#------------------------------------------------
#Accepts columns that have only 2 different values
#Changing one of the values to 0 and the second to 1
def two_option_col(colName,val1,val2, df):



    index = df.index.tolist()
    col=list(df[colName])
    indexCol = 0
    for i in index:
        if col[indexCol] == val1:
            df[colName].at[i] = 0
        elif col[indexCol] == val2:
            df[colName].at[i] = 1
        indexCol += 1





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
#A method that checks whether the cell is empty- no information
def isNaN(val):
    return val != val or val=='' or val=='?'
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
'''
def recognizeColByNum(file,col_num):

    header=list(file.columns.values)
    colName=header[col_num]

    return colName'''
#-------------------------------
#method that replaces an average nan value of the same column
def replaceNaN(file,colName,avg):
    #=file.iloc[0,col_num]
    #print(col_num," ",col_name," ")
    col=list(file[colName]) # create list from row
    rowNum=0
    #colName= recognizeColByNum(file,col_num)

    for i in col:
        if isNaN(i):
            #col = list(file[col_name])
            file[colName].at[rowNum] = avg

        rowNum+=1

#-------------------------------

#A method that normalizes specific col in the file
def normalization(file, col_num):
    colName = file.iloc[0, col_num]
    col = list(file[colName])
    #col = contain_col(col_num, file)  # create list from row
    average,line_count,flagNaN=averageCol(col)
    if flagNaN==1:
        replaceNaN(file,colName, average)
        col = list(file[colName])
        #col = contain_col(col_num, file)  # create list from row
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
    #colName = recognizeColByNum(file, col_num)
    file[colName]=normalization_col#Updating the column to be normalized


# -------------------------------



#A method that normalizes all cols in the file
def normalizationAll(file):
    colNum=file.shape[1]
    for colIndex in range(2,colNum-1):  # run on the num of cols WITHOUT V_ONE
        #colLine = contain_col(colIndex, file)
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