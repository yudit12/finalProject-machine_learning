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
def split_col_data(col_name,df):
     length= df.shape[0]
     col_num =df.columns.get_loc(col_name)
     col = contain_col(col_num, df)

     if col_name=='ChestPain':

         for i in range(length):
             if  col[i]=='typical':
                 df['typical'].at[i]=1
                 df['asymptomatic'].at[i]=0
                 df['nonanginal'].at[i]=0
                 df['nontypical'].at[i]=0
             elif col[i]=='asymptomatic':
                 df['typical'].at[i] = 0
                 df['asymptomatic'].at[i] = 1
                 df['nonanginal'].at[i] = 0
                 df['nontypical'].at[i] = 0
             elif col[i] == 'nonanginal':
                 df['typical'].at[i] = 0
                 df['asymptomatic'].at[i] = 0
                 df['nonanginal'].at[i] = 1
                 df['nontypical'].at[i] = 0
             elif col[i] == 'nontypical':
                 df['typical'].at[i] = 0
                 df['asymptomatic'].at[i] = 0
                 df['nonanginal'].at[i] = 0
                 df['nontypical'].at[i] = 1
     if col_name =='Thal':
         for i in range(length):
             if col[i] == 'fixed':
                 df['fixed'].at[i] = 1
                 df['normal'].at[i] = 0
                 df['reversable'].at[i] = 0
             elif isNaN(col[i]):
                 df['fixed'].at[i] = np.nan
                 df['normal'].at[i] = np.nan
                 df['reversable'].at[i] = np.nan
             elif col[i] == 'normal':
                 df['fixed'].at[i] = 0
                 df['normal'].at[i] = 1
                 df['reversable'].at[i] = 0
             elif col[i] == 'reversable':
                 df['fixed'].at[i] = 0
                 df['normal'].at[i] = 0
                 df['reversable'].at[i] = 1
     if col_name =='result':
             for i in range(length):
                 if col[i] == ' <=50K':#Less than 50k or equal
                     df['result'].at[i] = 0
                 elif col[i] == ' >50K':
                     df['result'].at[i] = 1#more than 50k


# -------------------------------