# this program reads data from a CSV containing Redfin listings for Pittsburgh, PA
# outlier elimination and data normalization are handled outside of this class 

import pandas as pd
import numpy as np

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

def fixIllogicals(df):
    dfIllogicallsDropped = df.copy()
    dfIllogicallsDropped = dfIllogicallsDropped[dfIllogicallsDropped['home_lotSize'] < 125000] # one record has lot size of 165310200... illogical 
    dfIllogicallsDropped = dfIllogicallsDropped[dfIllogicallsDropped['home_yrBuilt'] > 1620] # one record has home built in 1600s when settlement began around 1700s
    dfIllogicallsDropped.reset_index(drop=True, inplace=True) 
    return dfIllogicallsDropped

# create a binary feature for if a home has a HOA or not
# based on data cleaning function, approximately 95% of values are missing
# if a home has a HOA fee, this may help the regression prediction
def fixHOAFeature(df):
    dfHOA = df.copy()
    dfHOA['home_HOA'] = np.where(dfHOA['home_HOA'].isnull(), 0, 1)
    return dfHOA

# main function to call other methods
def main(redfinDataFile):
    df = importRedfinData(redfinDataFile)
    df = retainColumns(df)
    renameColumns(df)
    df = castDataTypes(df)
    df = fixHOAFeature(df)
    df = fixIllogicals(df)
    return df

# control program execution flow
if __name__ == '__main__':
    df = main("Data/redfin_sample.csv")
    print(df.describe().apply(lambda s: s.apply('{0:,.0f}'.format)).T , '\n')
    print(df.head(), '\n')