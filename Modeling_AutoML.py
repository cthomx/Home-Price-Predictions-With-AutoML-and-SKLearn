from autogluon.tabular import TabularPredictor
import pandas as pd

def performAutoGluonModelRun(train_data, X_test, y_test):
    label = 'PRICE'  # name of target variable to predict 
    save_path = 'AutoGluonModels/'  # where to store trained models

    # train auto ml model
    # source: https://www.analyticsvidhya.com/blog/2021/10/beginners-guide-to-automl-with-an-easy-autogluon-example/#h2_10
    # source: https://auto.gluon.ai/stable/tutorials/tabular_prediction/tabular-indepth.html
    predictor = TabularPredictor(label=label, eval_metric='mean_absolute_error').fit(
        train_data = train_data,
        num_gpus=1,  # Grant 1 gpu for the Tabular Predictor
        time_limit=800, 
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

    # make predictions on the test data set using the best model from autogluon
    pred_test_autogluon = predictor.predict(X_test)

    # limit results to feasible range
    yHat = pd.DataFrame(pred_test_autogluon)
    yHat.rename(columns={'PRICE':'yHat'}, inplace = True)
    yActual = pd.DataFrame(y_test)
    yActual.rename(columns={'PRICE':'yActual'}, inplace = True)
    yHat.reset_index(inplace=True, drop=True)
    yActual.reset_index(inplace=True, drop=True)
    testResults = pd.concat([yHat, yActual], axis=1)
    testResultsLimited = testResults.loc[(testResults['yActual'] > 100000) & (testResults['yActual'] < 500000)]

    return pred_test_autogluon, testResultsLimited