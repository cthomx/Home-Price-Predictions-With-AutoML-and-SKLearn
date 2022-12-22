import Data_Import_Redfin as rdi 
import Data_Import_Census as cdi 
import Data_Imputation as di
import Data_Staging as ds
import Evaluation_ModelPerformance as emp
import Modeling_SKLearn as msl
import Modeling_AutoML as mam

# set up Redfin data file
redfinDataFrameInit = rdi.main("Data/redfin_full.csv")

# set up combined census data + redfin data
df = cdi.main(redfinDataFrameInit)

# impute values
df = di.main(df)

# test train split 
X_train, X_test, y_train, y_test, train_data = ds.main(df)

# run AutoGluon Models
predictorAutoGluon = mam.main(train_data, X_test, y_train)

# make predictions on the test data set using the best model from autogluon
pred_test_autogluon = predictorAutoGluon.predict(X_test)

# get evaluation
emp.computeMetricsForRegression(pred_test_autogluon, 'AutoGluon', 'testing', y_test)

# run SK Learn Models
bestModel_SKLearn = msl.main(X_train, y_train)

# predict on test set 
pred_test_SKLearn = bestModel_SKLearn.predict(X_test)

# get evaluation
emp.computeMetricsForRegression(pred_test_SKLearn, bestModel_SKLearn.best_estimator_, 'testing', y_test)