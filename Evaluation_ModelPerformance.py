# function that computes the error metrics 

import numpy as np
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns

def computeMetricsForRegression(model_name, model_name_string,test_type, y_type): 
    
    RMSE = int(np.sqrt(mean_squared_error(y_type,model_name)))
    MAE = int(mean_absolute_error(y_type,model_name))
    R2 = r2_score(y_type, model_name)
    
    print('\n',model_name_string,' results ', test_type, '\n')
    print("RMSE: ", f"{RMSE:,}")
    print("MAE:  ", f"{MAE:,}")
    print("R^2:  ", f"{R2:,.2f}")
    print()
    
    # source: https://www.datacourses.com/evaluation-of-regression-models-in-scikit-learn-846/
    fig, ax = plt.subplots()
    ax.scatter(model_name, y_type, edgecolors=(0, 0, 1))
    ax.plot([y_type.min(), y_type.max()], [y_type.min(), y_type.max()], 'r--', lw=3)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.ticklabel_format(style='plain')
    plt.show()
