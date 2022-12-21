import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import Redfin_Data_Import as rdi 
import Census_Data_Import as cdi 

# generates a missing data report 
def getMissingReport(df):
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})
    missing_value_df = missing_value_df.sort_values(by=['percent_missing'], ascending = False)
    print('\n', missing_value_df.to_markdown(), '\n')

# helper function to impute missing values 
def fillInMissing(fillType, df, column, categoricalColumn="", yVariable=""):
    dfImputed = df.copy()
    if fillType == 1:
          dfImputed[column] = df[column].fillna(df[column].median())
          return dfImputed 
    elif fillType == 2:
        dfImputed[column] = df[column].fillna(df[column].mode())
        return dfImputed
    elif fillType == 3:
         dfImputed[column] = df[column].fillna(df[column].mean())
         return dfImputed
    elif fillType == 4: # class level continious imputation
        dfImputed[column] = df[column].fillna(df.groupby(categoricalColumn)[column].transform('mean'))
        return dfImputed
    elif fillType == 5: # class level categorical imputation
        dfImputed[column] = df[column].fillna(df.groupby(yVariable)[column].transform('mode'))
        return dfImputed
    else:
        print('Error, trace callabck logs')

# impute missing values with reasonable estimates based on either mean, median, mode, or class level
def imputeValues(df):
    dfFilled = df.copy()
    dfFilled = fillInMissing(1, dfFilled, 'home_sqft')
    dfFilled = fillInMissing(1, dfFilled, 'home_lotSize')
    dfFilled = fillInMissing(1, dfFilled, 'home_yrBuilt')
    dfFilled = fillInMissing(1, dfFilled, 'home_baths')
    return dfFilled

def main(df):
    dfCleaned = df.copy()
    getMissingReport(dfCleaned)
    dfCleaned = imputeValues(dfCleaned)
    getMissingReport(dfCleaned)
    return dfCleaned

# control program execution flow
if __name__ == '__main__':
    # set up Redfin data file
    redfinDataFrameInit = rdi.main("Data/redfin_sample.csv")

    # set up combined census data + redfin data
    df = cdi.main(redfinDataFrameInit)

    # combine with census data
    df = main(df)

    # inspect dataframe 
    print(df.describe().apply(lambda s: s.apply('{0:,.2f}'.format)).T , '\n')
    print(df.head(),'\n')