import pandas as pd
import numpy as np
import glob
import tabulate
import ast
import re
from pandas.core.common import SettingWithCopyWarning
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error, classification_report,accuracy_score,precision_recall_fscore_support
from scipy import stats
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

input_folder = 'C:/Users/olgaf/Documents/Thesis/logistics_for_ML/SAM3 total cycle/'
data = pd.read_csv(rf'{input_folder}Dataset_to_predict.csv')


def predict_regression(model, X, y, parameters=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # model = RandomForestRegressor(random_state=0)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    mse = round(mean_squared_error(y_test, y_hat), 4)
    # Calculate mean absolute percentage error (MAPE)
    errors = abs(y_hat - y_test)
    mape = round(np.mean(100 * (errors / np.maximum(y_test.values, errors.values))), 2)
    corr = round(stats.spearmanr(y_test, y_hat)[0], 2)
    # mape = mean_absolute_percentage_error(y_test, y_hat)
    metric_dict = {'model': 'RandomForestRegressor', 'target': y.name, 'mse': mse,'mape': mape, 'corr': corr}
    return metric_dict


def predict_classification(model, X, y, target_names, parameters=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    print(y_test.array, y_hat)
    print(classification_report(y_test.array, y_hat)) #, target_names=target_names))
    accuracy = accuracy_score(y_test, y_hat)
    precision,recall, fbeta_score, support = precision_recall_fscore_support(y_test, y_hat, zero_division="warn" )
    metric_dict = {'model': model, 'target': y.name, 'accuracy': accuracy, 'precision': precision, "recall": recall}
    return metric_dict


targets_class = ['Arrival', 'r_UB']
target_regr = ['r', 'x_b_sum', 'y_sum', 'x_w_sum']
features = [col for col in data.columns if col not in target_regr and col not in targets_class]
print(features)

regr_results = pd.DataFrame(columns = ['target','model','mse','mape','corr'])
class_results = pd.DataFrame(columns = ['target','model','accuracy','precision','recall'])

for col in target_regr[:-1]:
    prediction = predict_regression(RandomForestRegressor(), data.loc[:,features], data.loc[:, col])
    regr_results = regr_results.append(prediction, ignore_index=True)
    # print(regr_results.shape, '\n', tabulate.tabulate(regr_results.sample(1), headers=regr_results.columns, tablefmt='grid'))

arrival_encoding = {"LR": -1, "0": 0, "100": 1, "95": 2, "75": 3, "50": 4, "25": 5}
r_UB_encoding = {"0": 0, "25": 1, "50": 2, "75": 3, "95": 4, "100": 5}

class_results = class_results.append(predict_classification(RandomForestClassifier(), data.loc[:,features],data.loc[:,'Arrival'],list(arrival_encoding.keys())),ignore_index=True)
class_results = class_results.append(predict_classification(RandomForestClassifier(), data.loc[:,features],data.loc[:,'r_UB'].astype('int'),list(r_UB_encoding.keys())),ignore_index=True)

print(regr_results.shape, '\n', tabulate.tabulate(regr_results.sample(1), headers=regr_results.columns, tablefmt='grid'))
print(class_results.shape, '\n', tabulate.tabulate(class_results.sample(1), headers=class_results.columns, tablefmt='grid'))

results = pd.concat([regr_results, class_results], axis="columns")
results.to_csv(rf'{input_folder}All_prediction_results.csv')
