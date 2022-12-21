from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
import pandas as pd
import numpy as np

# one hot encode the censusTract joinKey 
def encodeVariables(df):
    # Get one hot encoding of columns B
    one_hot = pd.get_dummies(df['joinKey'], prefix='joinKey_')
    # Join the encoded df
    dfEncoded = df.join(one_hot)
    return dfEncoded

# drop featrues not relevatnt for prediction task or were previously encoded
def dropTargetedFeatures(df):
    dfDropped = df.copy()
    dfDropped = dfDropped.drop(columns=['censusTract', 'censusCounty', 'censusState', 'joinKey', 'home_latitude', 'home_longitude'], axis=1)
    return dfDropped

# create train test split and a 
def splitAndTransform(df):
    # set up X and Y
    X = df.loc[:, df.columns != 'home_price']
    y = df['home_price']

    # test train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=131)

    # set up scaler
    scaler = PowerTransformer(method='yeo-johnson')

    # exclude binary featuers
    scaling_columns = X.loc[: , X.columns[~X.columns.str.contains("home_HOA") & ~X.columns.str.startswith('joinKey_')]].columns 

    # train
    X_train[scaling_columns] = scaler.fit_transform(X_train[scaling_columns])
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    train_data = pd.concat([X_train, y_train], axis=1) # for auto-ml .. not scikit learn

    # test
    # do not fit the scaler on testing data, only transform it from training instances 
    X_test[scaling_columns] = scaler.transform(X_test[scaling_columns])
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    return X_train, X_test, y_train, y_test, train_data

# driver
def main(dfToLoad):
    df = encodeVariables(dfToLoad)
    df = dropTargetedFeatures(df)
    X_train, X_test, y_train, y_test, train_data = splitAndTransform(df)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    print(train_data.shape)
    print()
    return X_train, X_test, y_train, y_test, train_data

# control program execution flow
if __name__ == '__main__':
    import Data_Import_Redfin as rdi
    import Data_Import_Census as cdi 
    import Data_Imputation as di 
    
    # set up Redfin data file
    redfinDataFrameInit = rdi.main("Data/redfin_sample.csv")

    # set up combined census data + redfin data
    df = cdi.main(redfinDataFrameInit)

    # impute values
    df = di.main(df)

    # test train split 
    X_train, X_test, y_train, y_test, train_data = main(df)