from sklearn.model_selection import  train_test_split
from sklearn.impute import SimpleImputer
import numpy as np 
import pandas as pd

def retrieve_train_val(X, y, val_size = 0.2):

    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.2)
    median_imputer = SimpleImputer(missing_values=np.NaN,strategy='median')
    imputer = median_imputer.fit(train_X)
    v_train_X = imputer.transform(train_X)
    imputerval = median_imputer.fit(val_X)
    v_val_X = imputerval.transform(val_X)
    train_X = pd.DataFrame(v_train_X, columns = train_X.columns, index = train_X.index)
    val_X = pd.DataFrame(v_val_X, columns = val_X.columns,index = val_X.index)

    return train_X, val_X, train_y, val_y

def retrieve_test(X_test):
    
    median_imputer = SimpleImputer(missing_values=np.NaN,strategy='median')
    v_test_X = median_imputer.fit_transform(X_test)
    test_X = pd.DataFrame(v_test_X, columns=X_test.columns,index=X_test.index)

    return test_X
