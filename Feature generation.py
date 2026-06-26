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
import matplotlib.pyplot as plt
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

input_folder = 'C:/Users/olgaf/Documents/Thesis/logistics_for_ML/SAM3_final/'

def read_files(folder):
    ds_paths = glob.glob(folder)
    data = pd.read_csv(ds_paths[0], index_col=0)
    for file in ds_paths[1:]:
        dd = pd.read_csv(file, index_col=0)
        data = pd.concat([data,dd], axis=0)
        print(data.shape)
    print("Ratio of non-dublicate setups", data[~data.duplicated()].shape[0]/data.shape[0])
    data = data.drop_duplicates(keep="first")
    print(data.shape)
    return data


def prepare_inputs(df):
    # leave as it is
    input_1d = ['plan_horizon_TO', 'decision_time_Tb', 'ship_cost_c', 'increm_cost_z']
    # expand to two columns
    input_tuples = ['transp_time_Tr', 'base_repair_Tij', 'penalty_h', 'shortage_cost_pai']
    # take unique value
    input_multi_dim = ['items_fsl_S0tilde', 'items_warehouse_Stilde']
    # get digit
    cases = 'Cases'
    # targets = ['y_sum','x_b_sum','x_w_sum']
    df[cases] = df[cases].apply(lambda x: x.split(' ')[-1]).astype(int)
    df[input_1d] = df[input_1d].astype(int)

    input_tuples_upd = []
    for tup in input_tuples:
        df[tup] = df[tup].apply(lambda x: ast.literal_eval(x))
        col_name = tup.split('_')[-1]
        df[f'{col_name}_1'] = df[tup].apply(lambda x: x[0]).astype(int)
        df[f'{col_name}_2'] = df[tup].apply(lambda x: x[1]).astype(int)
        input_tuples_upd.append(f'{col_name}_1')
        input_tuples_upd.append(f'{col_name}_2')
    for col in input_multi_dim:
        df[col] = df[col].apply(lambda x: re.search(r'(\d)', x).group()).astype(int)
    columns_to_keep = input_1d + input_tuples_upd + input_multi_dim + [cases]
    df_cleaned = df[columns_to_keep]
    # print(df_cleaned.dtypes)
    # print(df_cleaned.shape, '\n', tabulate.tabulate(df_cleaned.sample(2), headers=df_cleaned.columns, tablefmt='grid'))
    return df_cleaned


def generate_features(df):
    df.loc[:,'c_z_ratio'] = np.where((df['ship_cost_c'] + df['increm_cost_z']) > 0,
                                       df['ship_cost_c']/(df['ship_cost_c'] + df['increm_cost_z']), -1)
    for fsl in [1,2]:
        if fsl==1:
            df.loc[:,f'pai_h_ratio_fsl_{fsl}'] = np.where((df[f'h_{fsl}'] + df[f'pai_{fsl}']) > 0,
                                                          df[f'h_{fsl}']/(df[f'h_{fsl}'] + df[f'pai_{fsl}']), -1)
            # close cycle == 0, distant cycle == 1
            df.loc[:,f'cycles_difference_fsl_{fsl}'] = np.where((df['plan_horizon_TO'] + df[f'Tr_{fsl}'] + df[f'Tij_{fsl}']) <= 5,
                                                                0, 1)
        elif fsl == 2:
            df.loc[:,f'pai_h_ratio_fsl_{fsl}'] = np.where((df[f'h_{fsl}'] + df[f'pai_{fsl}']) > 0,
                                                          df[f'h_{fsl}']/(df[f'h_{fsl}'] + df[f'pai_{fsl}']), -1)
            df.loc[:,f'cycles_difference_fsl_{fsl}'] = np.where((df['plan_horizon_TO'] + df[f'Tr_{fsl}'] + df[f'Tij_{fsl}']) <= 5, 0, 1)
    conditions = []
    for fsl in [1, 2]:
        conditions.append([(df[f'pai_{fsl}'] > df[f'h_{fsl}']+df['ship_cost_c']+df['increm_cost_z']),
                              (df[f'pai_{fsl}'] > df['increm_cost_z']+df['ship_cost_c']),
                              (df[f'pai_{fsl}'] >= df['increm_cost_z']) | (df[f'pai_{fsl}'] >= df['ship_cost_c']),
                              (df[f'pai_{fsl}'] < df['increm_cost_z']) & (df[f'pai_{fsl}'] < df['ship_cost_c'])])

    # high=4, moderate=3, fair=2, low =1
    bo_values = ["pai_gt_sum_hcz", "pai_gt_sum_cz", "pai_ge_c_or_z", "pai_lt_c_and_z"]
    df.loc[:,'Costs_relation_fsl1'] = np.select(conditions[0], bo_values)
    df.loc[:,'Costs_relation_fsl2'] = np.select(conditions[1], bo_values)
    # print(df.shape, '\n', tabulate.tabulate(df.sample(2), headers=df.columns, tablefmt='grid'))
    return df


def create_targets(data_targ):
    targets = ['x_b_sum','y_sum','x_w_sum']
    df = data_targ[targets].astype(int)
    df.loc[:, 'r'] = df.loc[:,'x_b_sum']/(df.loc[:,'x_w_sum'] + df.loc[:,'x_b_sum'])
    r_UB_conditions = [(df['r'] == 0),
                       (df['r'] <= 0.25),
                       (df['r'] <= 0.5),
                       (df['r'] <= 0.75),
                       # (df['r'] <= 0.95),
                       (df['r'] <= 1)]
    # r_UB_encoding = {0:0, 25:1, 50:2, 75:3, 95:4, 100:5}
    # r_UB_values = list(r_UB_encoding.values())
    r_UB_values = [0, 0.25, 0.5, 0.75, 1]
    df.loc[:, 'r_UB'] = np.select(r_UB_conditions, r_UB_values)
    arrival_conditions = [((df['x_w_sum'] == 0) & (df['x_b_sum'] != 0)),
                          (df['y_sum']/df['x_w_sum'] > 0.95),
                          (df['y_sum'] / df['x_w_sum'] > 0.75),
                          (df['y_sum'] / df['x_w_sum'] > 0.5),
                          (df['y_sum'] / df['x_w_sum'] > 0.25),
                          (df['y_sum'] / df['x_w_sum'] > 0.1),
                          (df['y_sum'] / df['x_w_sum'] > 0),
                            (df['y_sum'] / df['x_w_sum'] == 0)
                          ]

    # arrival_encoding = {"LR": -1, "y_sum_0": 0, "y > x_w": 1, "y < x_w": 2}
    # arrival_values = list(arrival_encoding.values()) #[-1, 0, 100, 95, 75, 50, 25]
    arrival_values = ["LR", "100", "95", "75", "50", "25", "10",  "0"]
    df.loc[:, 'wh_OI-bound_ratio'] = np.select(arrival_conditions, arrival_values)
    df.loc[:, 'r'] = df.loc[:,'r'].fillna(-1)
    df.loc[:, 'r_UB'] = df.loc[:, 'r_UB'].fillna(-1)
    return df.loc[:, ['x_b_sum','y_sum','x_w_sum', 'r_UB', 'wh_OI-bound_ratio']]


data = read_files(rf'{input_folder}SAM3_all cases*sample*.csv')
data.to_csv(rf"{input_folder}SAM3_all cases.csv")
# data = pd.read_csv(rf"{input_folder}SAM3_all cases_1e+4_sample.csv", index_col=0)
data = data.rename(columns={'Optimal #items (y_sum)':'y_sum'})
print(data.shape,'\n', tabulate.tabulate(data.sample(2), headers=data.columns, tablefmt='grid'))

data_updated = prepare_inputs(data)
features = generate_features(data_updated)
# print(features.shape, '\n', tabulate.tabulate(features.sample(2), headers=features.columns, tablefmt='grid'))
targets = create_targets(data)
dataset = pd.concat([features, targets], axis="columns")
print(dataset.shape, '\n', tabulate.tabulate(dataset.sample(2), headers=dataset.columns, tablefmt='grid'))
# dataset.to_csv(rf"{input_folder}Dataset_to_predict2411.csv", index=False)

dataset.to_csv(rf"{input_folder}Dataset_to_predict_v2.csv", index=False)


