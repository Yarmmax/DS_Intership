# linear algebra
import numpy as np
# working with data in table structures
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.2f' % x)
# data visualization
import seaborn as sns
import matplotlib.pyplot as plt
# working with files
import sys
import os
from pathlib import Path
import csv
import io
# to off warnings
import warnings
warnings.filterwarnings('ignore')
# validation schema
import time
import sklearn
from sklearn.model_selection import train_test_split
from datetime import timedelta, datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# metrics  calculation
from sklearn.metrics import (
    mean_squared_error as mse_lib,
    mean_absolute_error as mae_lib,
    mean_absolute_percentage_error as mape_lib,
    r2_score as r2_lib
)
from permetrics.regression import RegressionMetric

# models
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBRegressor
import catboost as cb


# Validation schema creation

"""
    The following indexes will be used:

        from train_data:
          date_block_num
          shop_id
          item_category_id
          item_id
          item_cnt_month

        from test_data:
            shop_id',
            'item_id',
            'ID'

    Concept:
        Apply expanding window validation (except last month - target of competition)
        Monthly predictions
"""


class Validation:
    def __init__(self,
                 train_data,
                 submission_data,
                 submission_example,
                 metrics=['RMSE', 'MAE', 'MAPE', 'sMAPE', 'R2'],
                 n_splits=5,
                 model=RandomForestRegressor(max_depth=1, n_estimators=1, random_state=42, n_jobs=-1),
                 params=None,
                 check_nans=True,
                 dropna=False,
                 check_infs=True,
                 validation_schema_plot=False
                 ):
        self.train_data = train_data
        self.submission_data = submission_data
        self.submission_example = submission_example
        self.metrics = metrics
        self.n_splits = n_splits
        self.model = model
        self.params = params
        self.tscv = TimeSeriesSplit(n_splits=self.n_splits)

        # Check data for valid columns
        assert set([
            'date_block_num',
            'shop_id',
            'item_category_id',
            'item_id',
            'item_cnt_month'
        ]).issubset(train_data.columns), \
            "Invalid data"

        assert set([
            'shop_id',
            'item_category_id',
            'item_id',
        ]).issubset(submission_data.columns), \
            "Invalid data"

        assert set([
            'shop_id',
            'item_id',
            'ID'
        ]).issubset(submission_example.columns), \
            "Invalid data"

        # Check for valid variables
        if dropna:
            self.train_data = self.train_data.dropna()
            self.submission_data = self.submission_data.dropna()

        if check_nans:
            assert self.train_data.isna().sum().sum() == 0, 'Train data have NaNs'
            assert self.submission_data.isna().sum().sum() == 0, 'Test data have NaNs'

        if check_infs:
            assert np.isfinite(self.train_data).sum().sum() != 0, 'Train data have Infs'
            assert np.isfinite(self.submission_data).sum().sum() != 0, 'Test data have Infs'

            # Сheck for sorting by timeseries data
        amount_of_unsorted_rows = len(self.train_data) - (self.train_data.date_block_num.diff().fillna(0) >= 0).sum()
        if amount_of_unsorted_rows != 0:
            print(
                f"Data is not sorted by time ({amount_of_unsorted_rows} rows), it will be further sorted automatically")
            self.train_data = self.train_data.sort_values(by=['date_block_num'])

        # Validation process visualisation
        if validation_schema_plot:
            split_history = {
                'Step': ["Step " + str(i) for i in range(1, self.n_splits + 1)],
                'Train Data': [],
                'Validation Data': []
            }
            for train, val in self.tscv.split(self.train_data[['item_id', 'shop_id', 'item_cnt_month']]):
                split_history['Train Data'].append(train.max() - train.min())
                split_history['Validation Data'].append(val.max() - val.min())
            df = pd.DataFrame(split_history)
            print(df)
            sns.set(style="whitegrid")
            plt.figure(figsize=(6, 5))
            sns.barplot(x='Step', y='Train Data', data=df, color='skyblue', label='Train Data')
            sns.barplot(x='Step', y='Validation Data', data=df, color='salmon', label='Validation Data',
                        bottom=df['Train Data'])
            plt.title('Distribution of data for training and validation')
            plt.xlabel('Validation step')
            plt.ylabel('Data quantity')
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.show()

    def train_test_split(self):
        """
        Split on train/test data
        where test data will contain records by last month of input train_data
        """

        last_month = self.train_data.date_block_num.max()
        test_data = self.train_data[self.train_data.date_block_num == last_month]
        train_data = self.train_data[self.train_data.date_block_num != last_month]
        return train_data, test_data

    def calculate_metrics(self, y_pred, y_true):
        """
        Return metrics of regression calculated on fitted model
        """

        metrics = {}

        metrics['RMSE'] = mse_lib(y_true=y_true, y_pred=y_pred, squared=True)
        metrics['MAE'] = mae_lib(y_true=y_true, y_pred=y_pred)
        metrics['MAPE'] = mape_lib(y_true=y_true, y_pred=y_pred)
        metrics['sMAPE'] = 100 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
        metrics["R2"] = r2_lib(y_true=y_true, y_pred=y_pred)

        return metrics

    def predict_submission(self, predictions_by_ID=True):
        """
            Return target predictions in accordance with submission example
            Target month: November 2015
        """
        X_train, y_train = self.train_data.drop(columns=['item_cnt_month']), \
            self.train_data.item_cnt_month
        X_sub = self.submission_data

        # train model
        model = self.model
        model.fit(X_train, y_train)
        if predictions_by_ID:
            result = X_sub.join(pd.DataFrame(index=X_sub.index, data=model.predict(X_sub.values), \
                                              columns=['item_cnt_month'])) \
                [['item_id', 'shop_id', 'item_cnt_month']]. \
                merge(self.submission_example, on=['shop_id', 'item_id'], how='right') \
                .drop_duplicates(['item_id', 'shop_id'])[['ID', 'item_cnt_month']].sort_values(by='ID')
            result.item_cnt_month = result.item_cnt_month.clip(0, 20).fillna(0)
            return result
        else:
            return model.predict(X_sub)

    def predict_test(self):
        """
         Return predictions based on test dataset
        """
        # Split data step (train/test)

        train_data, test_data = self.train_test_split()

        # Data selection
        X_train, y_train = train_data.drop(columns=['item_cnt_month']), \
            train_data.item_cnt_month
        X_test, y_test = test_data.drop(columns=['item_cnt_month']), \
            test_data.item_cnt_month

        # Train model
        model = self.model
        model.fit(X_train, y_train)

        # Evaliation test step

        y_pred_test = model.predict(X_test)

        # Return metrcis from test
        return pd.DataFrame.from_dict(self.calculate_metrics(y_pred=y_pred_test, y_true=y_test), orient='index',
                                      columns=['Test metrics'])

    def validate(self):

        """
        Implementation of validation using an expanding window
        """

        eval_report = {}
        train_errors = []
        val_errors = []
        is_boost = False

        # Build dataframe with metrics information
        metric_values = pd.MultiIndex.from_product([["step" + str(i) for i in range(1, self.n_splits + 1)],
                                                    ['train', 'validation']],
                                                   names=['Steps', 'Train/Validation/Test'])
        metrics = ['RMSE', 'MAE', 'MAPE', 'sMAPE', 'R2']
        metrics_info = pd.DataFrame('-', metric_values, metrics)

        # Define model
        rng = np.random.RandomState(42)
        model = self.model

        # Split data step (train/test)

        train_data, test_data = self.train_test_split()

        # Split train to X and y
        X = train_data.drop(columns='item_cnt_month')
        y = train_data[['item_id', 'shop_id', 'item_cnt_month']]

        # Evaluation loop
        step = 0
        for train, val in self.tscv.split(y):

            # Initialize steps and timer
            step += 1
            ts = time.time()

            # Split data step (train/validation)
            y_train, y_val = y.iloc[train].item_cnt_month, y.iloc[val].item_cnt_month
            X_train, X_val = X.iloc[train], X.iloc[val]

            # Train step
            if isinstance(model, sklearn.ensemble._forest.RandomForestRegressor):
                model = self.model
                model.fit(X_train, y_train)

            else:
                is_boost = True

                if isinstance(model, cb.core.CatBoostRegressor):
                    # Split
                    train_data = cb.Pool(X_train, label=y_train)
                    valid_data = cb.Pool(X_val, label=y_val)

                    # Train
                    model.fit(train_data, eval_set=valid_data)

                    # Get error report
                    evals_result = model.get_evals_result()
                    train_error = evals_result['learn']['RMSE']
                    val_error = evals_result['validation']['RMSE']

                if isinstance(model, XGBRegressor):
                    model = self.model
                    model.fit(X_train, y_train, eval_metric="rmse", eval_set=[(X_train, y_train), (X_val, y_val)],
                              verbose=25)

                    # Get error report
                    evals_result = model.evals_result()
                    train_error = evals_result['validation_0']['rmse']
                    val_error = evals_result['validation_1']['rmse']

                if isinstance(model, lgb.sklearn.LGBMRegressor):
                    assert self.params is not None, "params is None"
                    train_data = lgb.Dataset(X_train, label=y_train)
                    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val), (X_train, y_train)])

                    # Get error report
                    evals_result = model.evals_result_
                    train_error = evals_result['training']['l2']
                    val_error = evals_result['valid_0']['l2']

                train_errors.append(train_error)
                val_errors.append(val_error)

            # Get predictions
            y_pred_val = model.predict(X_val)
            y_pred_train = model.predict(X_train)

            # Calculate time required for step
            time_info = time.time() - ts

            # Metrics calucaltion step
            metrics_info.loc[("step" + str(step), 'validation'), :] = self.calculate_metrics(y_pred=y_pred_val,
                                                                                             y_true=y_val)
            metrics_info.loc[("step" + str(step), 'train'), :] = self.calculate_metrics(y_pred=y_pred_train,
                                                                                        y_true=y_train)

        # Evaliation test step
        # X_test, y_test = self.test_data.drop(columns=['item_cnt_month']), self.test_data.item_cnt_month
        # y_pred_test = model.predict(X_test)

        # Get metrcis from test

        # metrics_info.loc[("step1", "test"), :] = self.calculate_metrics(y_pred=y_pred_test, y_true=y_test)

        if is_boost:
            # Calculate mean values and std for train and validation error
            mean_train_errors = np.mean(train_errors, axis=0)
            std_train_errors = np.std(train_errors, axis=0)
            mean_val_errors = np.mean(val_errors, axis=0)
            std_val_errors = np.std(val_errors, axis=0)

            # Visualize learning curve with confidence intervals
            plt.figure(figsize=(10, 6))
            plt.plot(mean_train_errors, label='Average Train Error', color='blue')
            plt.plot(mean_val_errors, label='Average Validation Error', color='orange')
            plt.fill_between(range(len(mean_train_errors)), mean_train_errors - std_train_errors,
                             mean_train_errors + std_train_errors, color='lightblue', alpha=0.7)
            plt.fill_between(range(len(mean_val_errors)), mean_val_errors - std_val_errors,
                             mean_val_errors + std_val_errors, color='lightsalmon', alpha=0.3)
            plt.xlabel('Iterations')
            plt.ylabel('RMSE Error')
            plt.legend()
            plt.title(f'{type(model)} Average Training and Validation Error with Confidence Intervals')
            plt.show()

        return metrics_info