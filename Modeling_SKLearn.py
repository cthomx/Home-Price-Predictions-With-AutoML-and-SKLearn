# function to create a pipeline containing multiple models and grid search for hyperparameter tuning
# source: https://github.com/justmarkham/scikit-learn-tips/blob/master/notebooks/49_tune_multiple_models.ipynb

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

def setupMLPipeline(xtrain, ytrain):
    # set up models for evaluation
    reg1 = Ridge()
    reg2 = Lasso()
    reg3 = DecisionTreeRegressor()
    reg4 = RandomForestRegressor()
    reg5 = GradientBoostingRegressor()
    reg6 = MLPRegressor()

    # set up pipeline
    # it's intentiontal only 1 regressor is passed 
    # grid search will iterate through other regressors
    pipe = Pipeline([('regressor', reg1)])

    # create params dict for reg1
    params1 = {}
    params1['regressor__alpha'] = [1e-8,1e-3,1e-2,1,5,10,20,30,40,60,80,100,150,200,300,400,600,800]
    params1['regressor'] = [reg1]

    # create params dict for reg2
    params2 = {}
    params2['regressor__alpha'] = [1e-8,1e-3,1e-2,1,5,10,20,30,40,60,80,100,150,200,300,400,600,800]
    params2['regressor'] = [reg2]

    # create params dict for reg3
    params3 = {}
    params3['regressor__max_depth'] = [1,3,5,7,9,11,12]
    params3['regressor__min_samples_leaf'] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    params3['regressor'] = [reg3]

    # create params dict for reg3
    params4 = {}
    params4['regressor__n_estimators'] = [100, 200, 400]
    params4['regressor__max_depth'] = [3, 5, 7, 9] 
    params4['regressor__min_impurity_decrease'] = [0.0, 1.0, 3.0, 5.0, 7.0, 9.0]
    params4['regressor__max_features'] = ['sqrt','log2',None]
    params4['regressor'] = [reg4]

    # create params dict for reg3
    params5 = {}
    params5['regressor__learning_rate'] = [0.1, 0.2,0.3] 
    params5['regressor__max_depth'] = [3, 5, 7, 9] 
    params5['regressor'] = [reg5]

    # create params dict for reg3
    params6 = {}
    params6['regressor'] = [reg6]

    # create list of parameter dicts
    params = [params1, params2, params3, params4, params5, params6]

    # search every param combo within each dict
    grid = GridSearchCV(pipe, params)
    grid.fit(xtrain, ytrain)
    print(grid.best_params_)
    return grid

def main(xtrain, ytrain):
    return setupMLPipeline(xtrain, ytrain)

# control program execution flow
if __name__ == '__main__':
    import Data_Import_Redfin as rdi
    import Data_Import_Census as cdi 
    import Data_Imputation as di 
    import Data_Staging as ds
    
    # set up Redfin data file
    redfinDataFrameInit = rdi.main("Data/redfin_sample.csv")

    # set up combined census data + redfin data
    df = cdi.main(redfinDataFrameInit)

    # impute values
    df = di.main(df)

    # test train split 
    X_train, X_test, y_train, y_test, train_data = ds.main(df)

    # run SK Learn Models
    main(X_train, y_train)