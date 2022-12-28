from autogluon.tabular import TabularPredictor
import pandas as pd

def performAutoGluonModelRun(train_data, X_test, y_test):
    label = 'home_price'  # name of target variable to predict 
    save_path = 'AutoGluonModels/'  # where to store trained models

    # train auto ml model
    # source: https://www.analyticsvidhya.com/blog/2021/10/beginners-guide-to-automl-with-an-easy-autogluon-example/#h2_10
    # source: https://auto.gluon.ai/stable/tutorials/tabular_prediction/tabular-indepth.html
    predictor = TabularPredictor(label=label, path=save_path,eval_metric='mean_absolute_error').fit(
        train_data = train_data,
        num_gpus=1,  # Grant 1 gpu for the Tabular Predictor
        time_limit=200, 
        presets='best_quality',
    )

    # look at leaderboard 
    predictor.leaderboard(train_data, silent = True)

    # inspect results 
    results = predictor.fit_summary()
    print(results)

    # feature importance
    importance = predictor.feature_importance(feature_stage='original', data=pd.concat([X_test, y_test],axis=1))

    # view importance 
    print(importance)

    return predictor

def main(train_data, X_test, y_test):
    return performAutoGluonModelRun(train_data, X_test, y_test)

if __name__ == '__main__':
    import Data_Import_Redfin as rdi
    import Data_Import_Census as cdi 
    import Data_Imputation as di 
    import Data_Staging as ds
    import Evaluation_ModelPerformance as emp
    
    # set up Redfin data file
    redfinDataFrameInit = rdi.main("Data/redfin_sample.csv")

    # set up combined census data + redfin data
    df = cdi.main(redfinDataFrameInit)

    # impute values
    df = di.main(df)

    # test train split 
    X_train, X_test, y_train, y_test, train_data = ds.main(df)

    # run AutoGluon Models
    predictor = main(train_data, X_test, y_train)

    # make predictions on the test data set using the best model from autogluon
    pred_test_autogluon = predictor.predict(X_test)

    # get evaluation
    emp.computeMetricsForRegression(pred_test_autogluon, 'AutoGluon', 'testing', y_test)