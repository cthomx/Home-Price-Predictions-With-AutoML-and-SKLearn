# this program reads data from a CSV containing Redfin listings for Pittsburgh, PA
# outlier elimination and data normalization are handled outside of this class 

import pandas as pd

# import data
def importRedfinData(fileStr):
    return pd.read_csv(fileStr)

# dropping columns that may lead to data leakage outside of the target variable 'PRICE' 
def retainColumns(df):

    dfRetained = df[[
        'BEDS'
        ,'BATHS'
        ,'SQUARE FEET'	
        ,'LOT SIZE'
        ,'YEAR BUILT'
        ,'HOA/MONTH'
        ,'LATITUDE'
        ,'LONGITUDE'
        ,'PRICE']]

    return dfRetained

# conform naming convention
def renameColumns(df): 
     df.rename(columns={
        'BEDS': 'home_beds'
        ,'BATHS': 'home_baths'
        ,'SQUARE FEET': 'home_sqft'
        ,'LOT SIZE': 'home_lotSize'
        ,'YEAR BUILT': 'home_yrBuilt'
        ,'HOA/MONTH': 'home_HOA'
        ,'LATITUDE': 'home_latitude'
        ,'LONGITUDE': 'home_longitude'
        ,'PRICE': 'home_price'
     }, inplace=True)

# cast as integer or float
def castDataTypes(df):

    dfCasted = df.copy()

    # cast data types
    for i in dfCasted.columns.drop(['home_latitude', 'home_longitude']):
        try:
            dfCasted[i] = dfCasted[i].astype(int)
        except:
            dfCasted[i] = dfCasted[i].astype(float)
    
    return dfCasted

# main function to call other methods
def main(redfinDataFile):
    df = importRedfinData(redfinDataFile)
    df = retainColumns(df)
    renameColumns(df)
    df = castDataTypes(df)
    return df

# control program execution flow
if __name__ == '__main__':
    df = main("redfin_2022-12-20-20-35-47.csv")
    print(df.describe().T, '\n')
    print(df.head(), '\n')