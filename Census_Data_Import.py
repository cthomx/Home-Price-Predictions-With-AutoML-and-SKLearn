# functions to pull information relevant for a regression prediction task 
# this data will need to be cleaned and merged with other data sources

# import packages 
import pandas as pd
import numpy as np
import requests
import os
import censusgeocode as cg 

# register for census API https://api.census.gov/data/key_signup.html
# store key as environment variable 
CENSUS_API_KEY = os.getenv('CENSUS_API_KEY')

# get the census tract by latitude and longitude based on redfin data
def getCensusGeoData(xInit, yInit): 
    initResult = cg.coordinates(x=xInit, y=yInit).get('Census Tracts')[0]
    tract = initResult.get('TRACT') 
    county = initResult.get('COUNTY')    
    state = initResult.get('STATE') 
    return tract, county, state

# enrich redfin data containing latitude and longitude with Census data
def assignCensusGeoData(df):

    dfrf = df.copy() # dataframe with redfin 

    # get target census tract from redfin data
    dfrf[['censusTract', 'censusCounty', 'censusState']] = dfrf.apply(lambda row: getCensusGeoData(row.home_longitude, row.home_latitude), axis = 1, result_type="expand")
    
    # set up key to join in census data
    dfrf['joinKey'] = dfrf['censusState'] + dfrf['censusCounty'] + dfrf['censusTract']

    return dfrf 

# https://www.census.gov/data/developers/data-sets/acs-1year.html
# https://www.census.gov/content/dam/Census/data/developers/api-user-guide/api-guide.pdf
# representative map of census tract: https://www2.census.gov/geo/maps/DC2020/PL20/st42_pa/censustract_maps/c42003_allegheny/DC20CT_C42003.pdf
def getACSData(targetYear, targetCounty, targetState):
   
    # get web response
    webResponse=requests.get('https://api.census.gov/data/'+targetYear+'/acs/acs5?get=NAME,B01002_001E,B25109_001E,B25111_001E,B08134_001E,B08134_002E,B08134_003E,B08134_004E,B08134_005E,B08134_006E,B08134_007E,B15012_001E,B15012_009E,B15003_001E,B15003_023E,B15003_024E,B15003_025E,B19001_001E,B19001_002E,B19001_003E,B19001_004E,B19001_005E,B19001_006E,B19001_007E,B19001_008E,B19001_009E,B19001_010E,B19001_014E,B19001_015E,B19001_016E,B19001_017E,B19083_001E&for=tract:*&in=state:'+targetState+'+county:'+targetCounty+"&key="+CENSUS_API_KEY).json()
    
    # format web response
    df = pd.DataFrame.from_records(webResponse)
    df.columns = df.iloc[0] # enforce columns
    df = df[1:] # keep data but have properly formatted columns from index 0 
    df['joinKey'] = df['state'] + df['county'] + df['tract'] # set up join key to merge with geoData and redfin
    df.drop(columns = ['state', 'county', 'tract', 'NAME'], inplace=True)
    
    # set up variable list to type cast
    varList = ['B01002_001E','B25109_001E' ,'B25111_001E' 
               ,'B08134_001E','B08134_002E','B08134_003E','B08134_004E','B08134_005E','B08134_006E','B08134_007E' 
               ,'B15012_001E' ,'B15012_009E'
               ,'B15003_001E' ,'B15003_023E' ,'B15003_024E' ,'B15003_025E' ,'B19001_001E' 
               ,'B19001_002E' ,'B19001_003E' ,'B19001_004E' ,'B19001_005E' ,'B19001_006E' ,'B19001_007E' 
               ,'B19001_008E' ,'B19001_009E' ,'B19001_010E' ,'B19001_014E' ,'B19001_015E' ,'B19001_016E'
               ,'B19001_017E' ,'B19083_001E']
    
    # cast data types
    for i in varList:
        try:
            df[i] = df[i].astype(int)
        except:
            df[i] = df[i].astype(float)
            
    # fix null data sent from Census Bureau ... will a negative number 
    for i in varList: 
        try:
            df.loc[(df[i] < 0), i] = df[i].median()
        except Exception:
            pass # do nothing
    
    # rename columns
    df.rename(columns={
        'B01002_001E':'age_Median'
        ,'B25109_001E':'housing_OwnerOccupiedMedianValue'
        ,'B25111_001E':'renting_MedianRentValue'
        ,'B08134_001E':'commute_Total'
        ,'B08134_002E':'commute_LessThan10mins'
        ,'B08134_003E':'commute_10to14mins'
        ,'B08134_004E':'commute_15to19mins'
        ,'B08134_005E':'commute_20to24mins'
        ,'B08134_006E':'commute_26to29mins'
        ,'B08134_007E':'commute_30to34mins'
        ,'B15012_001E':'bachelors_Total'
        ,'B15012_009E':'bachelors_STEM'
        ,'B15003_001E':'education_Total'
        ,'B15003_023E':'education_MasterDegree'
        ,'B15003_024E':'education_ProfessionalDegree'
        ,'B15003_025E':'education_DoctorateDegree'
        ,'B19001_001E':'income_Total'
        ,'B19001_002E':'income_LessThan10K'
        ,'B19001_003E':'income_10Kto15K'
        ,'B19001_004E':'income_15Kto20K'
        ,'B19001_005E':'income_20Kto25K'
        ,'B19001_006E':'income_25Kto30K'
        ,'B19001_007E':'income_30Kto35K'
        ,'B19001_008E':'income_35Kto40K'
        ,'B19001_009E':'income_40Kto45K'
        ,'B19001_010E':'income_45Kto50K'
        ,'B19001_014E':'income_100Kto125K'
        ,'B19001_015E':'income_125Kto150K'
        ,'B19001_016E':'income_150Kto200K'
        ,'B19001_017E':'income_200KOrMore'
        ,'B19083_001E':'inequality_GiniIndex'
    }, inplace = True)
    
    # calculate percentages instead of raw numbers
    
    df['commute_pctLessThan34Mins'] = (df['commute_LessThan10mins'] + df['commute_10to14mins'] + df['commute_15to19mins'] + df['commute_20to24mins'] + df['commute_26to29mins'] + df['commute_30to34mins']) / df['commute_Total']
    df.drop(columns=['commute_LessThan10mins', 'commute_10to14mins', 'commute_15to19mins', 'commute_20to24mins', 'commute_26to29mins', 'commute_30to34mins', 'commute_Total'], inplace=True)
    
    df['bachelors_pctSTEM'] = df['bachelors_STEM'] / df['bachelors_Total']
    df.drop(columns=['bachelors_STEM', 'bachelors_Total'], inplace=True)
    
    df['education_pctAdvancedDegree'] = (df['education_MasterDegree'] + df['education_ProfessionalDegree'] + df['education_DoctorateDegree']) / df['education_Total']
    df.drop(columns=['education_MasterDegree', 'education_ProfessionalDegree', 'education_DoctorateDegree', 'education_Total'], inplace=True)
    
    df['income_pctBelow50K'] = (df['income_LessThan10K'] + df['income_10Kto15K'] + df['income_15Kto20K'] + df['income_20Kto25K'] + df['income_25Kto30K'] + df['income_30Kto35K'] + df['income_35Kto40K'] + df['income_40Kto45K'] + df['income_45Kto50K']) / df['income_Total']  
    df['income_pctAbove150K'] = (df['income_150Kto200K'] + df['income_200KOrMore']) / df['income_Total']
    df.drop(columns=['income_Total','income_LessThan10K','income_10Kto15K','income_15Kto20K','income_20Kto25K','income_25Kto30K','income_30Kto35K','income_35Kto40K','income_40Kto45K','income_45Kto50K','income_100Kto125K','income_125Kto150K','income_150Kto200K','income_200KOrMore'], inplace=True)
    
    return df

# get list of distince counties, will need to iterate over if numerous counties are passed
def getDistinceCounties(df):
    # check for numerous counties ... as the data scales so will this list
    distinctStateCounties = list(set(df['censusState'] + df['censusCounty']))
    return distinctStateCounties

# function to iterate over unique counties and append census data
def consolidateCensusOutput(uniqueCountyList):
    
    # set up list that will hold dataframe objects
    censusDFList = []

    # iterate and populate list with dataframe objects
    for i in range(len(uniqueCountyList)):
        print('Evaluating: ','2020', uniqueCountyList[i][2:], uniqueCountyList[i][:2])
        censusDFList.append(getACSData('2020', uniqueCountyList[i][2:], uniqueCountyList[i][:2]))
        
    # turn list of dataframe objects into single dataframe since indexes will be shared
    dfcensus = pd.concat(censusDFList)

    return dfcensus

# main function to call other methods
def main(redfinDataFile):
    dfRedfin = assignCensusGeoData(redfinDataFrameInit) # assign tract and FIPS information
    uniqueCounties = getDistinceCounties(dfRedfin) # get list of unique counties
    dfcensus = consolidateCensusOutput(uniqueCounties) # get census data for each county at tract level
    combined = dfRedfin.merge(dfcensus, on='joinKey', how='left') # merge the census and redfin data together 
    return combined

# control program execution flow
if __name__ == '__main__':
    # set up Redfin data file
    import Redfin_Data_Import as rdi 
    redfinDataFrameInit = rdi.main("redfin_2022-12-20-20-35-47.csv")

    # combine with census data
    df = main(redfinDataFrameInit)
    print(df.describe().T,'\n')
    print(df.head(),'\n')