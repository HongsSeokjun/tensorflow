import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
학생csv = 'jena_홍석준_submit3.csv'

path1 = 'C:/STUDY25JUN/_data/kaggle/jena/'
path2 = 'C:/STUDY25JUN/_save/Keras56/'

datasets = pd.read_csv(path1+'jena_climate_2009_2016.csv', index_col=0)

y_정답 = datasets.iloc[-144:,-1]
print(y_정답)
print(y_정답.shape)

학생꺼 = pd.read_csv(path2+ 학생csv, index_col=0)
print(학생꺼)

def rmse(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('Rmse :', rmse(y_정답, 학생꺼))