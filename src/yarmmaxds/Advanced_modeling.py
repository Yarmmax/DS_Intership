# linear algebra
import numpy as np

#working with data in table structers
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

# to off warnings
import warnings
warnings.filterwarnings('ignore')

# validation schema
import time
from datetime import timedelta, datetime
from sklearn.model_selection import TimeSeriesSplit
from collections import defaultdict
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from scipy.stats import randint, uniform

# metrics  calculation
from sklearn.metrics import (
    mean_squared_error as mse_lib,
    mean_absolute_error as mae_lib,
    mean_absolute_percentage_error as mape_lib,
    r2_score as r2_lib
)
from permetrics.regression import RegressionMetric

# advanced modeling
from boruta import BorutaPy
import optuna
import shap
shap.initjs()
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK

# models
import random
import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
import catboost as cb

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
                 model=cb.CatBoostRegressor(iterations=35, verbose=10),
                 check_nans=True,
                 dropna=False,
                 check_infs=True,
                 plot=True,
                 test_size=0.2,
                 validation_schema_plot=False
                 ):
        self.train_data = train_data
        self.submission_data = submission_data
        self.submission_example = submission_example
        self.metrics = metrics
        self.n_splits = n_splits
        self.model = model
        self.plot = plot
        self.test_size = test_size
        if self.n_splits > 1:
            self.tscv = TimeSeriesSplit(n_splits=self.n_splits)

        # Check data for valid columns
        assert set([
            'date_block_num',
            'shop_id',
            'item_id',
            'item_cnt_month'
        ]).issubset(train_data.columns), \
            "Invalid data"

        assert set([
            'shop_id',
            'item_id',
        ]).issubset(submission_data.columns), \
            "Invalid data"

        assert set([
            'shop_id',
            'item_id',
            'ID'
        ]).issubset(submission_example.columns), \
            "Invalid data"

        # Check for valid variables and drop nans if necessary
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

        # Split to X and y
        # self.X = self.train_data.drop(columns='item_cnt_month')
        # self.y = self.train_data[['item_id', 'shop_id', 'item_cnt_month']]

        # Validation process visualisation
        if validation_schema_plot:
            if self.plot and n_splits > 1:
                split_history = {
                    'Step': ["Step" + str(i) for i in range(1, self.n_splits + 1)],
                    'Train Data': [],
                    'Validation Data': []
                }
                self.tscv = TimeSeriesSplit(n_splits=self.n_splits)
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

    def validate(self, predictions_by_ID=True, type="score", plot=False):
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

        if self.n_splits > 1:

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

            # Extract metric for validation score
            metric = metrics_info.loc(axis=0)[:, 'validation']['RMSE']

        elif self.n_splits == 1:
            # Predict test and calculate score on it
            metrics_info = self.predict_test()
            metric = metrics_info.loc['RMSE', :]

        # Return result of validation
        if type == "report":
            return metrics_info
        elif type == "score":
            return np.asarray(metric).mean()


class Pipeline:
    def __init__(self,
                 train_data,
                 submission_data,
                 submission_example,
                 metrics=['rmse'],
                 model=cb.CatBoostRegressor(iterations=35, silent=True),
                 check_nans=True,
                 dropna=False,
                 check_infs=True,
                 feature_importance_layer=False,
                 hyperparametr_optimization_layer=False,
                 params=None,
                 optimizer="Grid",
                 selection_sample_size=None,
                 explainability_layer=False,
                 error_analysis_layer=False,
                 optimizer_iterations=20
                 ):
        self.train_data = train_data
        self.submission_data = submission_data
        self.submission_example = submission_example
        self.metrics = metrics
        self.model = model
        self.fitted_model = None
        self.predictions = None
        self.params = params
        self.optimal_hyperparametres = None
        self.important_features = None
        self.important_features_index = None
        self.selection_sample_size = selection_sample_size
        self.optimizer = optimizer
        self.optimizer_iterations = optimizer_iterations
        self.__feature_importance_layer__ = feature_importance_layer
        self.__hyperparametr_optimization_layer__ = hyperparametr_optimization_layer
        self.__explainability_layer__ = explainability_layer
        self.__error_analysis_layer__ = error_analysis_layer
        self.selected_train_data = None
        self.selected_X = None
        self.selected_test_data = None

        # Check data for valid columns
        assert set([
            'date_block_num',
            'shop_id',
            'item_id',
            'item_cnt_month'
        ]).issubset(train_data.columns), \
            "Invalid data"

        assert set([
            'shop_id',
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

        # Split to X and y
        self.X = self.train_data.drop(columns='item_cnt_month')
        self.y = self.train_data[['item_id', 'shop_id', 'item_cnt_month']]

        # Сolumns required for validation
        self.train_validation_features = [
            'date_block_num',
            'shop_id',
            'item_id',
            'item_cnt_month'
        ]
        self.test_validation_features = [
            'shop_id',
            'item_id',
        ]
        self.test_X_features = [
            'shop_id',
            'item_id',
            'date_block_num'
        ]

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

    # Predict sales for target month (November 2015)
    def predict_target(self):
        """
            Return target predictions in accordance with submission example
        """

        if self.__feature_importance_layer__:
            train_features = self.important_features.copy()
            train_features.extend(self.train_validation_features)
            train_data = self.train_data[train_features]

            test_features = self.important_features.copy()
            test_features.extend(self.test_validation_features)
            test_data = self.test_data[test_features]

            submission_example = self.submission_example
        else:
            train_data = self.train_data
            test_data = self.test_data
            submission_example = self.submission_example

        X_train, y_train = train_data.drop(columns=['item_cnt_month']), train_data.item_cnt_month.clip(0, 20)
        X_test = test_data
        X_test['date_block_num'] = 34

        if self.optimal_hyperparametres:
            model = cb.CatBoostRegressor(**self.optimal_hyperparametres, verbose=35)
        elif self.params:
            model = cb.CatBoostRegressor(**self.params, verbose=35)
        else:
            model = self.model
        model.fit(X_train, y_train)

        # Get feature importance info and sort it
        feature_importance = model.get_feature_importance(prettified=True)
        feature_importance = feature_importance.sort_values(by='Importances', ascending=True)

        # Visualize importatnt features
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['Feature Id'], feature_importance['Importances'])
        plt.title('Feature Importances')
        plt.xlabel('Importance')
        plt.show()

        # Save model fitting
        self.fitted_model = model.copy()
        self.predictions = model.predict(X_test.values)

        result = X_test.join(pd.DataFrame(index=X_test.index, data=self.predictions, \
                                          columns=['item_cnt_month'])) \
            [['item_id', 'shop_id', 'item_cnt_month']]. \
            merge(self.submission_example, on=['shop_id', 'item_id'], how='right') \
            .drop_duplicates(['item_id', 'shop_id'])[['ID', 'item_cnt_month']].sort_values(by='ID')
        result.item_cnt_month = result.item_cnt_month.clip(0, 20).fillna(0)
        return result

    # Predict sales for target month (November 2015)
    def predict_test(self):
        """
            Return calculated metrics and test predictions in accordance with test dataset
        """
        # Split data step (train/test)

        train_data, test_data = self.train_test_split()

        if self.__feature_importance_layer__:
            train_features = self.important_features.copy()
            train_features.extend(self.train_validation_features)
            train_data = train_data[train_features]

            test_features = self.important_features.copy()
            test_features.extend(self.test_validation_features)
            test_data = test_data[test_features]

        X_train, y_train = train_data.drop(columns=['item_cnt_month']), train_data.item_cnt_month.clip(0, 20)
        X_test, y_test = test_data.drop(columns=['item_cnt_month']), test_data.item_cnt_month.clip(0, 20)

        if self.optimal_hyperparametres:
            model = cb.CatBoostRegressor(**self.optimal_hyperparametres, verbose=35)
        elif self.params:
            model = cb.CatBoostRegressor(**self.params, verbose=35)
        else:
            model = self.model
        model.fit(X_train, y_train)

        # Get feature importance info and sort it
        feature_importance = model.get_feature_importance(prettified=True)
        feature_importance = feature_importance.sort_values(by='Importances', ascending=True)

        # Visualize important features
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['Feature Id'], feature_importance['Importances'])
        plt.title('Feature Importances')
        plt.xlabel('Importance')
        plt.show()

        # Save model fitting
        self.fitted_model = model.copy()

        y_pred_test = model.predict(X_test.values)

        return sns.histplot(y_pred_test), pd.DataFrame.from_dict(
            self.calculate_metrics(y_pred=y_pred_test, y_true=y_test), orient='index', columns=['Test metrics'])

    def feature_importance_layer(self, selector="Boruta"):
        sample_size = self.selection_sample_size
        if sample_size is None:
            sample_size = self.train_data.shape[0]
        if selector == "Boruta":
            # Select sample of data
            X = self.train_data.dropna().drop(columns='item_cnt_month')[:sample_size]
            y = self.train_data.dropna()[:sample_size].item_cnt_month
            np.int = np.int_
            np.float = np.float_
            np.bool = np.bool_

            # Init selector
            feat_selector = BorutaPy(RandomForestRegressor(max_depth=7, n_estimators=50),
                                     n_estimators='auto',
                                     verbose=2,
                                     max_iter=20,
                                     random_state=42,
                                     )

            # Fit selector
            feat_selector.fit(X.values, y.values)

            # Extract usefull features
            self.important_features_index = feat_selector.support_
            self.important_features = self.X.columns[feat_selector.support_].tolist()

            # Save info about usefull/useless features
            feature_importance_report = {
                "important_columns": self.train_data.drop(columns=['item_cnt_month']) \
                                         .iloc[:, feat_selector.support_].columns,
                "unimportant_columns": self.train_data.drop(columns=['item_cnt_month']) \
                                           .iloc[:, ~feat_selector.support_].columns,
                "feature_importance_scores": feat_selector.ranking_
            }

            return feature_importance_report

    def hyperparametr_optimization_layer(self, optimizer="Grid"):

        if self.__feature_importance_layer__:
            train_features = self.important_features.copy()
            train_features.extend(self.train_validation_features)
            train_data = self.train_data[train_features]

            test_features = self.important_features.copy()
            test_features.extend(self.test_validation_features)
            test_data = self.test_data[test_features]

            submission_example = self.submission_example
        else:
            train_data = self.train_data
            test_data = self.test_data
            submission_example = self.submission_example

        print(f"Count of using features {train_data.shape[1]}")

        if optimizer == "Optuna":
            # Optimization fucntion
            def objective(trial):
                params = {
                    "iterations": trial.suggest_int("iterations", 10, 250),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                    "depth": trial.suggest_int("depth", 1, 5),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 2, log=True),
                    "subsample": trial.suggest_float("subsample", 0.05, 1.0),
                    "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
                }

                model = Validation(
                    train_data,
                    test_data,
                    submission_example,
                    n_splits=5,
                    model=cb.CatBoostRegressor(**params, silent=True),
                    check_nans=False,
                    plot=False
                )
                rmse = model.validate()
                return rmse

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=self.optimizer_iterations)

            print('Best hyperparameters:', study.best_params)
            print('Best RMSE:', study.best_value)

            return study.best_params

        if optimizer == "Hyperopt":
            # Optimization fucntion
            def objective(params):
                params['depth'] = int(params['depth'])

                # Model creation
                model = Validation(
                    train_data,
                    test_data,
                    submission_example,
                    n_splits=5,
                    model=cb.CatBoostRegressor(**params, silent=True),
                    check_nans=False,
                    plot=False
                )

                # Calculate validation error
                return {'loss': model.validate(plot=False), 'status': STATUS_OK}

            # Define space for hyperparameters tuning

            space = {
                "iterations": hp.randint("iterations", 10, 250),
                'depth': hp.quniform('depth', 2, 5, 1),
                'learning_rate': hp.loguniform('learning_rate', -3, 0),
                'l2_leaf_reg': hp.loguniform('l2_leaf_reg', 1e-3, 2),
                "subsample": hp.uniform("subsample", 0.05, 1.0),
                "colsample_bylevel": hp.uniform("colsample_bylevel", 0.05, 1.0),

            }

            # Trials for recording optimization process
            trials = Trials()

            # TPE optimizer
            best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=self.optimizer_iterations, trials=trials)

            # Print best parameters
            print("Best parameters:")
            print(best)

            # Get best params
            best_params = {**space_eval(space, best)}

            return best_params

    def explainability_layer(self, ind_pred_to_expl=5, pred_with_high_sales=True):
        if self.__feature_importance_layer__:
            train_features = self.important_features.copy()
            train_features.extend(['item_cnt_month'])
            # train_features.extend(self.train_validation_features)
            train_data = self.train_data[train_features]
            submission_data = self.submission_data

            test_features = self.important_features.copy()
            test_features.extend(['shop_id', 'item_id'])
            test_data = self.test_data[test_features]
        else:
            train_data = self.train_data
            test_data = self.test_data
            submission_data = self.submission_data

        if self.optimal_hyperparametres:
            model = cb.CatBoostRegressor(**self.optimal_hyperparametres, silent=True)
        elif self.params:
            model = cb.CatBoostRegressor(**self.params, silent=True)
        else:
            model = self.model

        model.fit(train_data.drop(columns='item_cnt_month'), train_data.item_cnt_month)

        explainer = shap.Explainer(model)
        shap_values = explainer(test_data)

        # visualize SHAP values aggregated by first all predictions

        shap.summary_plot(shap_values, plot_type='bar')
        shap.summary_plot(shap_values, plot_type='violin')

        # collect SHAP values by item_id/shop_id combinations
        shap_info = {}
        shap_values.data = np.nan_to_num(shap_values.data, nan=0)
        if self.__feature_importance_layer__:
            for i in range(len(shap_values)):
                shap_info[tuple([int(float_num) for float_num in shap_values[i].data[-2:]])] = shap_values[i]
        else:
            for i in range(len(shap_values)):
                shap_info[tuple([int(float_num) for float_num in shap_values[i].data[1:3]])] = shap_values[i]

        # select shops/items combinations
        submission_examples = []
        for i in range(submission_data.shape[0]):
            submission_examples.append(tuple(submission_data.iloc[i][1:3]))

        # create dict with ID <-> shop_id, item_id connection
        ID_by_shop_item = {'ID': [int(i) for i in submission_data.ID],
                           'shop_id': [int(i) for i in submission_data.shop_id],
                           'item_id': [int(i) for i in submission_data.item_id]}
        df = pd.DataFrame(ID_by_shop_item)
        ID_to_shop_item = dict(zip(df['ID'], zip(df['shop_id'], df['item_id'])))

        # select predictions with high sales
        preds = \
        test_data.join(pd.DataFrame(index=test_data.index, data=model.predict(test_data), \
                                            columns=['item_cnt_month'])) \
            [['item_id', 'shop_id', 'item_cnt_month']]. \
            merge(submission_data, on=['shop_id', 'item_id'], how='right') \
            .drop_duplicates(['item_id', 'shop_id'])[['ID', 'item_cnt_month']].sort_values(by='ID').fillna(0)
        predictions_with_high_sales = preds.sort_values(by=['item_cnt_month'], ascending=False)[:100]

        if pred_with_high_sales == True:
            # visualize SHAP values for random submission predictions with high sales
            counter = 0
            while counter < ind_pred_to_expl:
                random_ID_with_high_sales = ID_to_shop_item \
                    [predictions_with_high_sales.ID.sample().values[0]]
                if random_ID_with_high_sales in shap_info.keys():
                    counter += 1
                    shap.plots.waterfall(shap_info[random_ID_with_high_sales])

        else:
            # visualize SHAP values for random submission predictions
            counter = 0
            while counter < ind_pred_to_expl:
                random_ID = ID_to_shop_item[int(preds.iloc[random.randint(0, preds.shape[0] - 1)].ID)]
                if random_ID in shap_info.keys():
                    counter += 1
                    shap.plots.waterfall(shap_info[random_ID])

    def error_analysis_layer(self):
        error_analysis_report = []

        X = self.X
        y = self.y.item_cnt_month

        # spit traind data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # learn model on train
        model = self.model
        model.fit(X_train, y_train)

        # predict y_test
        y_pred = self.model.predict(X_test)

        # calculate error
        errors_info = pd.DataFrame({"y_pred": y_pred,
                                    "y_test": y_test,
                                    "error": np.abs(y_pred - y_test),
                                    }).join(X_test)

        # adding 2 features for error analysis (one for small dynamic and another for big)
        errors_info['y_pred'] = round(errors_info.y_pred)
        errors_info[['y_pred', 'y_test']] = errors_info[['y_pred', 'y_test']].replace(0, 1)
        errors_info["error_type_1"] = np.abs(errors_info.y_pred / errors_info.y_test)
        errors_info["error_type_2"] = np.abs(errors_info.y_test / errors_info.y_pred)

        # k is the boundary separation coefficient based on the ratio of the predicted to the origin
        k = 10
        high_error_predictions = errors_info.sort_values(by=['error'], ascending=False).head(100)
        small_dynamic_is_poorly_predicted = errors_info[errors_info.error_type_1 > k].sort_values(by=['error_type_1'],
                                                                                                  ascending=False)
        big_target_is_poorly_predicted = errors_info[errors_info.error_type_2 > k].sort_values(by=['error_type_2'],
                                                                                               ascending=False)
        percentage_of_big_small_poor_predictions = (errors_info[errors_info.error_type_1 > k].shape[0] \
                                                    + errors_info[errors_info.error_type_2 > k].shape[0]) \
                                                   / X_test.shape[0] * 100
        items_most_often_errors_occur = {"small_dynamic": small_dynamic_is_poorly_predicted.groupby('item_id')['error'] \
            .count()[small_dynamic_is_poorly_predicted.groupby('item_id')['error'] \
                         .count().sort_values(ascending=False) > 10].sort_values(ascending=False).index,
                                         "big_dynamic": big_target_is_poorly_predicted.groupby('item_id')['error'] \
                                             .count()[big_target_is_poorly_predicted.groupby('item_id')['error'] \
                                                          .count().sort_values(ascending=False) > 10].sort_values(
                                             ascending=False).index
                                         }

        error_analysis_report = {
            "high_error_predictions": high_error_predictions,
            "poorly_small_dynamic": small_dynamic_is_poorly_predicted,
            "poorly_high_dynamic": big_target_is_poorly_predicted,
            "percentage_of_big_small_poor_predictions": f"{percentage_of_big_small_poor_predictions} %",
            "items_most_often_errors_occur": items_most_often_errors_occur
        }

        return error_analysis_report

    def evaluate(self):
        if self.__feature_importance_layer__:
            feature_importance_report = self.feature_importance_layer()
        if self.__hyperparametr_optimization_layer__:
            self.optimal_hyperparametres = self.hyperparametr_optimization_layer(optimizer=self.optimizer)
            predictions = self.predict_target()
        else:
            predictions = self.predict_target()

        if self.__explainability_layer__:
            self.explainability_layer()

        if self.__error_analysis_layer__:
            error_analysis_report = self.error_analysis_layer()

        if self.__feature_importance_layer__ and self.__error_analysis_layer__:
            return feature_importance_report, error_analysis_report, predictions
        elif self.__feature_importance_layer__:
            return feature_importance_report, predictions
        elif self.__error_analysis_layer__:
            return error_analysis_report, predictions
        else:
            return predictions