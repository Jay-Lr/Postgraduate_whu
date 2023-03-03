
"""
    readrawdata:补充连续时间序列;计算各个变量的日均值
                提取23：00的土壤墒情数据
    dataForAllLayers：获取全部层的输入数据
    dataForEachLayer: 获取单层的输入数据
    输入：日均气象变量和23：00土壤湿度数据，土壤组成和容重数据
    模型：
        lstm_model
        svr_model
        rf_model

"""
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as skm  # 评价指标计算
from scipy.optimize import leastsq
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import scipy.interpolate as spi
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import RepeatVector, TimeDistributed
import math
from sklearn.metrics import r2_score
from tensorflow.keras.models import load_model
from tensorflow import keras
from sklearn.utils import shuffle

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

import xgboost as xgb
import lightgbm as lgb
import pickle
import joblib  # 保存机器学习代码

# -*- coding : utf-8-*-
# coding:unicode_escape

# 中文字体显示
plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False




# 绘制各个变量的趋势图变化
def plot_features(dataset):
    # 创建一个包含八个子图的图像，每个子图对应一个变量
    plt.figure(figsize=(16, 12), dpi=300)
    for i in range(1, len(dataset.columns)):  # 除去时间列
        plt.subplot(len(dataset.columns), 1, i)
        feature_name = dataset.columns[i]
        plt.plot(dataset[feature_name])
        plt.rcParams.update({'font.size': 5})
        plt.title(feature_name, y=0, size=6)
        plt.grid(linestyle='--', alpha=0.5)
        plt.xticks(fontsize=5)
        plt.yticks(fontsize=5)

    plt.tight_layout()
    plt.show()

# --------------------------------------------------------------模型-----------------------------------------------------------------

def svr_model(train, path):
    """
    :param train: 训练数据，[trainx,trainy]
    :param path: 模型保存路径
    :return:
    """
    [train_x, train_y] = train
    train_y = np.ravel(train_y)  # 改成(shapes,)
    # 参数设置
    parameters = {'kernel': ['rbf'], 'gamma': np.logspace(-5, 2, num=6, base=2.0),
                  'C': np.logspace(-5, 5, num=11, base=2.0)}
    # 网格搜索：选择十折交叉验证
    """
        refit：默认为True，程序将会以交叉验证训练集得到的最佳参数，重新对所有可用的训练集与测试集进行
        n_jobs: 并行数，int：个数,-1：跟CPU核数一致, 1:默认值。
    """
    svr = SVR()
    grid_search = GridSearchCV(svr, parameters, cv=10, n_jobs=4, scoring='neg_mean_squared_error')
    # SVR模型训练
    """
        grid.fit( train_x, train_y )：运行网格搜索
        grid_scores_：给出不同参数情况下的评价结果 (旧版本)
        cv_results_: 
        best_params_：描述了已取得最佳结果的参数的组合
        best_score_：最佳估计量的平均交叉验证得分
    """
    grid_search.fit(train_x, train_y)
    # 输出最终的参数
    print(grid_search.cv_results_)
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    # SVR模型保存
    joblib.dump(grid_search, path)
    return


def rf_model(train, path, level):
    [train_x, train_y] = train
    train_y = np.ravel(train_y)  # 改成(shapes,)
    if (level == 1):
        param_grid = {
            'criterion': ['mse'],
            'n_estimators': [54],  # 决策树个数
            'max_depth': [9],  # 深度：这里是森林中每棵决策树的深度
            # 'max_features': np.arange(8,30,2), # 每棵决策树使用的变量占比
            # 'min_samples_split': [4, 8, 12, 16]  # 叶子的最小拆分样本量
        }
    elif (level == 2):
        param_grid = {
            'criterion': ['mse'],
            'n_estimators': [55],  # 决策树个数
            'max_depth': [6],  # 深度：这里是森林中每棵决策树的深度
        }
    elif (level == 3):
        param_grid = {
            'criterion': ['mse'],
            'n_estimators': [41],  # 决策树个数
            'max_depth': [8],  # 深度：这里是森林中每棵决策树的深度
        }
    elif (level == 4):
        param_grid = {
            'criterion': ['mse'],
            'n_estimators': [45],  # 决策树个数
            'max_depth': [8],  # 深度：这里是森林中每棵决策树的深度
        }
    elif (level == 5):
        param_grid = {
            'criterion': ['mse'],
            'n_estimators': [35],  # 决策树个数
            'max_depth': [9],  # 深度：这里是森林中每棵决策树的深度
        }
    elif (level == 6):
        param_grid = {
            'criterion': ['mse'],
            'n_estimators': [38],  # 决策树个数
            'max_depth': [8],  # 深度：这里是森林中每棵决策树的深度
        }
    elif (level == 7):
        param_grid = {
            'criterion': ['mse'],
            'n_estimators': [48],  # 决策树个数
            'max_depth': [7],  # 深度：这里是森林中每棵决策树的深度
        }
    elif (level == 8):
        param_grid = {
            'criterion': ['mse'],
            'n_estimators': [52],  # 决策树个数
            'max_depth': [9],  # 深度：这里是森林中每棵决策树的深度
        }
    elif (level == 9):
        param_grid = {
            'criterion': ['mse'],
            'n_estimators': [45],  # 决策树个数
            'max_depth': [8],  # 深度：这里是森林中每棵决策树的深度
        }
    elif (level == 10):
        param_grid = {
            'criterion': ['mse'],
            'n_estimators': [45],  # 决策树个数
            'max_depth': [8],  # 深度：这里是森林中每棵决策树的深度
        }

    rfr = RandomForestRegressor()
    rfr_cv = GridSearchCV(estimator=rfr, param_grid=param_grid,
                          scoring='neg_mean_squared_error', cv=5)  # neg_mean_squared_error
    rfr_cv.fit(train_x, train_y)

    print(rfr_cv.best_params_)  # 查看梯度优化给出的最佳参数，如果参数落在了给定范围（param_grid）的边界上，需要调整参数的范围
    print(rfr_cv.best_score_)  # 0.940034

    # importances = list(rfr_cv.feature_importances_)
    # print(importances)
    # joblib.dump(rfr_cv, path)
    return


def lightgbm_model(train, path, level):
    # 训练集
    [train_x, train_y] = train
    train_y = np.ravel(train_y)
    # lightgbm模型参数设置
    cv_params = {
        # 'n_estimators':np.arange(100,500,10),
        # 'max_depth': [3, 5, 7, 9,],
        # 'num_leaves': np.arange(2, 20, 2),
    }
    if (level == 1):
        params = {
            'boosting_type': 'gbdt',  # 使用的树的模型：梯度提升决策树（GBDT）
            'objective': 'regression',  # 目标函数 使用L2正则项的回归模型（默认值)
            'metric': 'mse',  # 评估函数
            'n_estimators': 110,  # 分类器数量
            'learning_rate': 0.1,  # 学习速率  [0.01, 0.015, 0.025, 0.05, 0.1]
            'max_depth': 7,  # 树的最大深度 [3, 5, 6, 7, 9, 12, 15, 17, 25]
            'num_leaves': 14,  # 一般设置为(0, 2^max_depth - 1]的一个数值
            'colsample_bytree': 0.6,  # 建树的特征选择比例 [0.6, 0.7, 0.8, 0.9, 1]
            'subsample': 1,  # 建树的样本采样比例 [0.6, 0.7, 0.8, 0.9, 1]
            'bagging_freq': 2,  # k 意味着每 k 次迭代执行bagging  [2, 4, 5, 6, 8]
            'reg_alpha': 0,
            'reg_lambda': 0,
        }
    elif (level == 2):
        params = {
            'boosting_type': 'gbdt',  # 使用的树的模型：梯度提升决策树（GBDT）
            'objective': 'regression',  # 目标函数 使用L2正则项的回归模型（默认值)
            'metric': 'mse',  # 评估函数
            'n_estimators': 230,  # 分类器数量
            'learning_rate': 0.1,  # 学习速率  [0.01, 0.015, 0.025, 0.05, 0.1]
            'max_depth': 4,  # 树的最大深度 [3, 5, 6, 7, 9, 12, 15, 17, 25]
            'num_leaves': 16,  # 一般设置为(0, 2^max_depth - 1]的一个数值
            'colsample_bytree': 0.9,  # 建树的特征选择比例 [0.6, 0.7, 0.8, 0.9, 1]
            'subsample': 0.9,  # 建树的样本采样比例 [0.6, 0.7, 0.8, 0.9, 1]
            'bagging_freq': 2,  # k 意味着每 k 次迭代执行bagging  [2, 4, 5, 6, 8]
            'reg_alpha': 0,
            'reg_lambda': 36,
        }
    elif (level == 3):
        params = {
            'boosting_type': 'gbdt',  # 使用的树的模型：梯度提升决策树（GBDT）
            'objective': 'regression',  # 目标函数 使用L2正则项的回归模型（默认值)
            'metric': 'mse',  # 评估函数
            'n_estimators': 290,  # 分类器数量
            'learning_rate': 0.04,  # 学习速率  [0.01, 0.015, 0.025, 0.05, 0.1]
            'max_depth': 9,  # 树的最大深度 [3, 5, 6, 7, 9, 12, 15, 17, 25]
            'num_leaves': 16,  # 一般设置为(0, 2^max_depth - 1]的一个数值
            'colsample_bytree': 0.9,  # 建树的特征选择比例 [0.6, 0.7, 0.8, 0.9, 1]
            'subsample': 0.9,  # 建树的样本采样比例 [0.6, 0.7, 0.8, 0.9, 1]
            'bagging_freq': 2,  # k 意味着每 k 次迭代执行bagging  [2, 4, 5, 6, 8]
            'reg_alpha': 0.1,
            'reg_lambda': 0,
        }
    elif (level == 4):
        params = {
            'boosting_type': 'gbdt',  # 使用的树的模型：梯度提升决策树（GBDT）
            'objective': 'regression',  # 目标函数 使用L2正则项的回归模型（默认值)
            'metric': 'mse',  # 评估函数
            'n_estimators': 230,  # 分类器数量
            'learning_rate': 0.03,  # 学习速率  [0.01, 0.015, 0.025, 0.05, 0.1]
            'max_depth': 7,  # 树的最大深度 [3, 5, 6, 7, 9, 12, 15, 17, 25]
            'num_leaves': 14,  # 一般设置为(0, 2^max_depth - 1]的一个数值
            'colsample_bytree': 0.9,  # 建树的特征选择比例 [0.6, 0.7, 0.8, 0.9, 1]
            'subsample': 0.9,  # 建树的样本采样比例 [0.6, 0.7, 0.8, 0.9, 1]
            'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging  [2, 4, 5, 6, 8]
            'reg_alpha': 0,
            'reg_lambda': 0.6,
        }
    elif (level == 5):
        params = {
            'boosting_type': 'gbdt',  # 使用的树的模型：梯度提升决策树（GBDT）
            'objective': 'regression',  # 目标函数 使用L2正则项的回归模型（默认值)
            'metric': 'mse',  # 评估函数
            'n_estimators': 250,  # 分类器数量
            'learning_rate': 0.03,  # 学习速率  [0.01, 0.015, 0.025, 0.05, 0.1]
            'max_depth': 4,  # 树的最大深度 [3, 5, 6, 7, 9, 12, 15, 17, 25]
            'num_leaves': 8,  # 一般设置为(0, 2^max_depth - 1]的一个数值
            'colsample_bytree': 0.9,  # 建树的特征选择比例 [0.6, 0.7, 0.8, 0.9, 1]
            'subsample': 0.9,  # 建树的样本采样比例 [0.6, 0.7, 0.8, 0.9, 1]
            'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging  [2, 4, 5, 6, 8]
            'reg_alpha': 0.03,
            'reg_lambda': 0.6,
        }
    elif (level == 6):
        params = {
            'boosting_type': 'gbdt',  # 使用的树的模型：梯度提升决策树（GBDT）
            'objective': 'regression',  # 目标函数 使用L2正则项的回归模型（默认值)
            'metric': 'mse',  # 评估函数
            'n_estimators': 260,  # 分类器数量
            'learning_rate': 0.03,  # 学习速率  [0.01, 0.015, 0.025, 0.05, 0.1]
            'max_depth': 3,  # 树的最大深度 [3, 5, 6, 7, 9, 12, 15, 17, 25]
            'num_leaves': 5,  # 一般设置为(0, 2^max_depth - 1]的一个数值
            'colsample_bytree': 0.9,  # 建树的特征选择比例 [0.6, 0.7, 0.8, 0.9, 1]
            'subsample': 0.9,  # 建树的样本采样比例 [0.6, 0.7, 0.8, 0.9, 1]
            'bagging_freq': 4,  # k 意味着每 k 次迭代执行bagging  [2, 4, 5, 6, 8]
            'reg_alpha': 0.03,
            'reg_lambda': 2,
        }
    elif (level == 7):
        params = {
            'boosting_type': 'gbdt',  # 使用的树的模型：梯度提升决策树（GBDT）
            'objective': 'regression',  # 目标函数 使用L2正则项的回归模型（默认值)
            'metric': 'mse',  # 评估函数
            'n_estimators': 445,  # 分类器数量
            'learning_rate': 0.02,  # 学习速率  [0.01, 0.015, 0.025, 0.05, 0.1]
            'max_depth': 3,  # 树的最大深度 [3, 5, 6, 7, 9, 12, 15, 17, 25]
            'num_leaves': 6,  # 一般设置为(0, 2^max_depth - 1]的一个数值
            'colsample_bytree': 0.9,  # 建树的特征选择比例 [0.6, 0.7, 0.8, 0.9, 1]
            'subsample': 1,  # 建树的样本采样比例 [0.6, 0.7, 0.8, 0.9, 1]
            'bagging_freq': 2,  # k 意味着每 k 次迭代执行bagging  [2, 4, 5, 6, 8]
            'reg_alpha': 0,
            'reg_lambda': 0,
        }
    elif (level == 8):
        params = {
            'boosting_type': 'gbdt',  # 使用的树的模型：梯度提升决策树（GBDT）
            'objective': 'regression',  # 目标函数 使用L2正则项的回归模型（默认值)
            'metric': 'mse',  # 评估函数
            'n_estimators': 100,  # 分类器数量
            'learning_rate': 0.1,  # 学习速率  [0.01, 0.015, 0.025, 0.05, 0.1]
            'max_depth': 5,  # 树的最大深度 [3, 5, 6, 7, 9, 12, 15, 17, 25]
            'num_leaves': 12,  # 一般设置为(0, 2^max_depth - 1]的一个数值
            'colsample_bytree': 1,  # 建树的特征选择比例 [0.6, 0.7, 0.8, 0.9, 1]
            'subsample': 0.9,  # 建树的样本采样比例 [0.6, 0.7, 0.8, 0.9, 1]
            'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging  [2, 4, 5, 6, 8]
            'reg_alpha': 0,
            'reg_lambda': 0.63,
        }
    elif (level == 9):
        params = {
            'boosting_type': 'gbdt',  # 使用的树的模型：梯度提升决策树（GBDT）
            'objective': 'regression',  # 目标函数 使用L2正则项的回归模型（默认值)
            'metric': 'mse',  # 评估函数
            'n_estimators': 55,  # 分类器数量
            'learning_rate': 0.1,  # 学习速率  [0.01, 0.015, 0.025, 0.05, 0.1]
            'max_depth': 7,  # 树的最大深度 [3, 5, 6, 7, 9, 12, 15, 17, 25]
            'num_leaves': 22,  # 一般设置为(0, 2^max_depth - 1]的一个数值
            'colsample_bytree': 0.8,  # 建树的特征选择比例 [0.6, 0.7, 0.8, 0.9, 1]
            'subsample': 0.7,  # 建树的样本采样比例 [0.6, 0.7, 0.8, 0.9, 1]
            'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging  [2, 4, 5, 6, 8]
            'reg_alpha': 0,
            'reg_lambda': 0.63,
        }
    elif (level == 10):
        params = {
            'boosting_type': 'gbdt',  # 使用的树的模型：梯度提升决策树（GBDT）
            'objective': 'regression',  # 目标函数 使用L2正则项的回归模型（默认值)
            'metric': 'mse',  # 评估函数
            'n_estimators': 160,  # 分类器数量
            'learning_rate': 0.08,  # 学习速率  [0.01, 0.015, 0.025, 0.05, 0.1]
            'max_depth': 6,  # 树的最大深度 [3, 5, 6, 7, 9, 12, 15, 17, 25]
            'num_leaves': 14,  # 一般设置为(0, 2^max_depth - 1]的一个数值
            'colsample_bytree': 0.8,  # 建树的特征选择比例 [0.6, 0.7, 0.8, 0.9, 1]
            'subsample': 0.7,  # 建树的样本采样比例 [0.6, 0.7, 0.8, 0.9, 1]
            'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging  [2, 4, 5, 6, 8]
            'reg_alpha': 0,
            'reg_lambda': 0,
        }

    model = lgb.LGBMRegressor(**params)

    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=5,
                                 verbose=1, n_jobs=4)
    optimized_GBM.fit(train_x, train_y, verbose=False)

    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
    # 保存模型
    joblib.dump(optimized_GBM, path)
    return


def xgboost_model(train, path, level):
    [train_x, train_y] = train
    train_y = np.ravel(train_y)
    # 使用xgboost库的sklearn接口
    cv_params = {
        # 'n_estimators': np.arange(1, 750, 2)
    }
    if (level == 1):
        other_params = {
            'learning_rate': 0.1, 'n_estimators': 725,
            'max_depth': 6, 'min_child_weight': 3, 'seed': 0,
            'subsample': 0.8, 'colsample_bytree': 0.9,
            'gamma': 0.03, 'reg_alpha': 0, 'reg_lambda': 1
        }
    elif (level == 2):
        other_params = {
            'learning_rate': 0.03, 'n_estimators': 710,
            'max_depth': 4, 'min_child_weight': 3, 'seed': 0,
            'subsample': 0.8, 'colsample_bytree': 0.9,
            'gamma': 0.03, 'reg_alpha': 0, 'reg_lambda': 0
        }
    elif (level == 3):
        other_params = {
            'learning_rate': 0.03, 'n_estimators': 740,
            'max_depth': 3, 'min_child_weight': 6, 'seed': 0,
            'subsample': 1, 'colsample_bytree': 0.8,
            'gamma': 0.03, 'reg_alpha': 0, 'reg_lambda': 0
        }
    elif (level == 4):
        other_params = {
            'learning_rate': 0.1, 'n_estimators': 730,
            'max_depth': 3, 'min_child_weight': 2, 'seed': 0,
            'subsample': 1, 'colsample_bytree': 0.9,
            'gamma': 0.03, 'reg_alpha': 0, 'reg_lambda': 0
        }
    elif (level == 5):
        other_params = {
            'learning_rate': 0.1, 'n_estimators': 120,
            'max_depth': 10, 'min_child_weight': 6, 'seed': 0,
            'subsample': 0.9, 'colsample_bytree': 1,
            'gamma': 0.03, 'reg_alpha': 0, 'reg_lambda': 0
        }
    elif (level == 6):
        other_params = {
            'learning_rate': 0.1, 'n_estimators': 740,
            'max_depth': 9, 'min_child_weight': 4, 'seed': 0,
            'subsample': 0.9, 'colsample_bytree': 1,
            'gamma': 0.03, 'reg_alpha': 0, 'reg_lambda': 0
        }
    elif (level == 7):
        other_params = {
            'learning_rate': 0.1, 'n_estimators': 740,
            'max_depth': 8, 'min_child_weight': 8, 'seed': 0,
            'subsample': 0.9, 'colsample_bytree': 1,
            'gamma': 0.03, 'reg_alpha': 0, 'reg_lambda': 0
        }
    elif (level == 8):
        other_params = {
            'learning_rate': 0.1, 'n_estimators': 690,
            'max_depth': 8, 'min_child_weight': 4, 'seed': 0,
            'subsample': 0.9, 'colsample_bytree': 0.8,
            'gamma': 0.004, 'reg_alpha': 0, 'reg_lambda': 0

        }
    elif (level == 9):
        other_params = {
            'learning_rate': 0.1, 'n_estimators': 740,
            'max_depth': 4, 'min_child_weight': 6, 'seed': 0,
            'subsample': 0.9, 'colsample_bytree': 0.8,
            'gamma': 0.004, 'reg_alpha': 0, 'reg_lambda': 0
        }
    elif (level == 10):
        other_params = {
            'learning_rate': 0.1, 'n_estimators': 740,
            'max_depth': 4, 'min_child_weight': 3, 'seed': 0,
            'subsample': 0.9, 'colsample_bytree': 0.8,
            'gamma': 0.002, 'reg_alpha': 0, 'reg_lambda': 0
        }

    model = xgb.XGBRegressor(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=5,
                                 verbose=1, n_jobs=4)
    optimized_GBM.fit(train_x, train_y)
    # res = optimized_GBM.cv_results_
    # test_score = res['mean_test_score']
    # fig = plt.figure(figsize=(5,5))
    # epochs = np.arange(1,750,2)
    # plt.plot(epochs, test_score, 'b', label='Training acc')  # 'bo'为画蓝色圆点，不连线
    # plt.legend()  # 绘制图例，默认在右上角
    # fig.savefig("accuracy.png", dpi=300)
    # print('每轮迭代运行结果:{0}'.format(optimized_GBM.cv_results_))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
    # joblib.dump(optimized_GBM, path)
    return

# ---------------------------------------------------------------------------------------------------------------------------------


# 划分训练，测试集
def split_dataset(data):
    '''
        该函数切分训练数据和测试数据  【576，100】
    '''
    # 打乱所有数据
    random.seed(5)
    data = np.array(data[1:561])
    index = [i for i in range(len(data))]
    random.shuffle(index)
    data = data[index]
    train, test = data[0:-164, 4:], data[-164:, 4:]
    return np.array(train), np.array(test)
# 对预测点5
def split_dataset_P5(data):
    '''
        该函数切分预测点5的训练数据和测试数据
    '''
    # 打乱所有数据
    data = np.array(data[1:348])
    random.seed(5)
    index = [i for i in range(len(data))]
    random.shuffle(index)
    data = data[index]
    train, test = data[0:-100, 4:], data[-100:, 4:]
    return np.array(train), np.array(test)


# 滑动窗口
def sliding_window(train, sw_width=1, n_out=1, in_start=0):
    # data = train.reshape((train.shape[0], train.shape[1]))  # 二维 [n,]
    X = train[:, 0:-10]
    # t时刻23：00土壤湿度为预测值
    Y = train[:, -10:]
    return np.array(X), np.array(Y)


# 划分每层数据并归一化
def dataForEachLayer(data, level):
    """
    :param x: 数组x
    :param y: 数组y
    :param level: 土壤层数
    :return:
    """
    random.seed(5)
    [x, y] = data
    index = [i for i in range(len(x))]
    random.shuffle(index)
    x = x[index]
    y = y[index]
    x1 = x[:, 0:16]  # 气象
    x_bd = x[:, 16 + level - 1]
    x_sand = x[:, 26 + level - 1]
    x_silt = x[:, 36 + level - 1]
    x_clay = x[:, 46 + level - 1]
    x_t_1 = x[:, 56 + level - 1]
    X = np.c_[x1, x_bd, x_sand, x_silt, x_clay, x_t_1]
    Y = y[:, level - 1]
    # 先划分再归一化
    data, scaler_y = scalerVariable(X, Y, 16, 20)
    return data, scaler_y


# 10层数据一起预测
def dataForAllLayers(train_x, train_y):
    # 打乱数据集
    index = [i for i in range(len(train_x))]
    random.seed(5)
    random.shuffle(index)
    train_x = train_x[index]
    train_y = train_y[index]

    # train数据归一化 MinMaxScaler只能用于一/二维数据,24:64指不需要归一化的列
    train, scaler_trainy = scalerVariable(train_x, train_y, 16, 20)
    # train, scaler_trainy = NormalizerVariable(train_x, train_y, 24, 64)
    return train, scaler_trainy


# 处理原始数据
def readrawData(pathdir, bulk_density, soilComp):
    """
        预测点1,2,3,4，分辨率为小时级，
        预测点 5,6，分辨率为分钟级，
        使用小时采样保证时间连续且插值
    """
    # 气象数据 首行为列名，第一列时间列为索引列 dataframe
    df = pd.read_csv(pathdir[0], header=0)  # encoding="gbk"
    df['时间'] = pd.to_datetime(df['时间'])
    df.set_index("时间", inplace=True)
    # 小于0的数据赋值为0
    df[df < 0] = 0
    # SMST数据
    df1 = pd.read_csv(pathdir[1], header=0)  # encoding="gbk"
    df1['时间'] = pd.to_datetime(df1['时间'])
    df1.set_index("时间", inplace=True)
    df1[df1 < 0] = 0
    # 检查时间序列是否连续(使用小时重采样实现)
    df0 = df.resample('H').mean().interpolate()
    df2 = df1.resample('H').mean().interpolate()
    # 保存时间连续，小时级别的数据
    df0.to_csv(path_or_buf=pathdir[2], index_label="时间", encoding='utf_8_sig')
    df2.to_csv(path_or_buf=pathdir[3], index_label="时间", encoding='utf_8_sig')

    # 计算气象因子的日均值  按时间重采样,计算变量日均值，label=right是指 索引保留右侧的取值，[0:00,23:00]
    df_day = df0.resample('D', label='left').mean().reset_index()
    # 气象因子，取t-1,t的滞后
    df0_lag1 = df_day.shift(1).add_suffix("_t-1")
    df0_lag0 = df_day.add_suffix("_t")
    meteo = pd.concat([df0_lag1, df0_lag0], axis=1)  # 按行拼接

    # 取出每天23:00的墒情数据  x指 行索引，其中，表头为0
    df_23h = pd.read_csv(pathdir[3], header=0, skiprows=lambda x: x > 0 and x % 24 != 0)
    # 仅保留需要数据
    df_23h = df_23h[['时间', '水分含量(%)-10', '水分含量(%)-20', '水分含量(%)-30', '水分含量(%)-40',
                     '水分含量(%)-50', '水分含量(%)-60', '水分含量(%)-70', '水分含量(%)-80', '水分含量(%)-90', '水分含量(%)-100']]
    # 取23点土壤湿度t-1的滞后
    df_23h_lay1 = df_23h.shift(1).add_suffix("_t-1")
    df_23h_lay0 = df_23h.add_suffix("_t")
    SM = pd.concat([df_23h_lay1, df_23h_lay0], axis=1)

    # 合并气象和墒情数据
    df_day_all = meteo.join(SM, rsuffix='_right')
    # 查看缺失值
    print(df_day_all.isna().sum())  # 无缺失值

    # 绘制变量变化趋势
    # plot_features(df_day)
    #  添加土壤容重数据
    df_day_all['bd10'], df_day_all['bd20'], df_day_all['bd30'], df_day_all['bd40'], df_day_all['bd50'], df_day_all[
        'bd60'], df_day_all['bd70'], df_day_all['bd80'], df_day_all['bd90'], df_day_all['bd100'] = bulk_density
    # 加入土壤组成数据
    df_day_all['sand10'], df_day_all['sand20'], df_day_all['sand30'], df_day_all['sand40'], df_day_all['sand50'], \
    df_day_all['sand60'], df_day_all['sand70'], df_day_all['sand80'], df_day_all['sand90'], df_day_all['sand100'], \
    df_day_all['silt10'], df_day_all['silt20'], df_day_all['silt30'], df_day_all['silt40'], df_day_all['silt50'], \
    df_day_all['silt60'], df_day_all['silt70'], df_day_all['silt80'], df_day_all['silt90'], df_day_all['silt100'], \
    df_day_all['clay10'], df_day_all['clay20'], df_day_all['clay30'], df_day_all['clay40'], df_day_all['clay50'], \
    df_day_all['clay60'], df_day_all['clay70'], df_day_all['clay80'], df_day_all['clay90'], df_day_all['clay100'] = soilComp

    print(df_day_all.columns.values)
    # 确保列名顺序一致 '土壤温度(℃)-120','土壤温度(℃)-150','水分含量(%)-120','水分含量(%)-150',
    #            '土壤温度(℃)-120_y', '土壤温度(℃)-150_y', '水分含量(%)-120_y', '水分含量(%)-150_y' 是预测点3 使用的
    order = [ '时间_t-1', '时间_t', '时间_t-1_right', '时间_t_right',
             '相对湿度(%)_t-1', '大气压力(hPa)_t-1', '风速(m/s)_t-1', '风向(°)_t-1', '雨量(mm)_t-1','当前太阳辐射强度(W/m²)_t-1',
             '累计太阳辐射量(MJ/m²)_t-1','空气温度(℃)_t-1',
             '相对湿度(%)_t', '大气压力(hPa)_t', '风速(m/s)_t', '风向(°)_t', '雨量(mm)_t', '当前太阳辐射强度(W/m²)_t',
             '累计太阳辐射量(MJ/m²)_t','空气温度(℃)_t',
             'bd10', 'bd20', 'bd30', 'bd40', 'bd50', 'bd60', 'bd70', 'bd80', 'bd90', 'bd100',
             'sand10', 'sand20', 'sand30', 'sand40', 'sand50', 'sand60', 'sand70', 'sand80', 'sand90', 'sand100',
             'silt10', 'silt20', 'silt30', 'silt40', 'silt50', 'silt60', 'silt70', 'silt80', 'silt90', 'silt100',
             'clay10', 'clay20', 'clay30', 'clay40', 'clay50', 'clay60', 'clay70', 'clay80', 'clay90', 'clay100',
             '水分含量(%)-10_t-1', '水分含量(%)-20_t-1', '水分含量(%)-30_t-1', '水分含量(%)-40_t-1',
             '水分含量(%)-50_t-1', '水分含量(%)-60_t-1', '水分含量(%)-70_t-1', '水分含量(%)-80_t-1', '水分含量(%)-90_t-1',
             '水分含量(%)-100_t-1',
             '水分含量(%)-10_t', '水分含量(%)-20_t', '水分含量(%)-30_t', '水分含量(%)-40_t',
             '水分含量(%)-50_t', '水分含量(%)-60_t', '水分含量(%)-70_t', '水分含量(%)-80_t', '水分含量(%)-90_t',
             '水分含量(%)-100_t',

             ]
    df_day_all = df_day_all[order]
    # 另存为csv
    df_day_all.to_csv(path_or_buf=pathdir[4], index_label="时间", encoding='utf-8_sig', index=False)


# 预测点3缺少70，90cm的土壤温湿度，多项式拟合土壤深度和温湿度，插值
def interpolate(path, savepath):
    df = pd.read_csv(path, header=0)
    # 温度数据 考虑到地表温湿度受外界条件影响较大，只是有地下数据
    x = [10, 20, 30, 40, 50, 60, 80, 100, 120, 150]
    x0 = [70, 90]
    ST = df.iloc[:, 2:12]
    SM = df.iloc[:, 12:]
    ST_y = np.empty(shape=[0, 2], dtype=float)
    SM_y = np.empty(shape=[0, 2], dtype=float)
    R_square = []
    for i in range(len(ST)):
        # 最小二乘拟合 ,多项式拟合  不行！！
        # p1=np.polyfit(x,ST.loc[i],2)
        # getmodel1 = np.poly1d(np.polyfit(x, ST.loc[i], 2))
        # yhat1 = np.polyval(p1, x0)
        # p2 = np.polyfit(x, SM.loc[i], 2)
        # getmodel2 = np.poly1d(np.polyfit(x, SM.loc[i], 2))
        # R_square.append([r2_score(ST.loc[i], getmodel1(x)),r2_score(SM.loc[i], getmodel2(x))])
        # yhat2 = np.polyval(p2, x0)
        # ST_y = np.append(ST_y, [yhat1], axis=0)
        # SM_y = np.append( SM_y,[yhat2], axis=0)  # 添加整行元素，axis=1添加整列元素
        # 三次样条插值
        STipo3 = spi.splrep(x, ST.loc[i], k=3)  # 源数据点导入，生成参数
        STiy3 = spi.splev(x0, STipo3)  # 根据观测点和样条参数，生成插值
        SMipo3 = spi.splrep(x, SM.loc[i], k=3)
        SMiy3 = spi.splev(x0, SMipo3)
        ST_y = np.append(ST_y, [STiy3], axis=0)
        SM_y = np.append(SM_y, [SMiy3], axis=0)  # 添加整行元素，axis=1添加整列元素
        # 三次样条 经过拟合点，无法用拟合点判断拟合效果
        # R_square.append([r2_score(np.array(ST.loc[i]),spi.splev(x, STipo3)), r2_score(np.array(SM.loc[i]), spi.splev(x,SMipo3))])
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        # ax1.plot(x, ST.loc[i], label='原数值')
        # ax1.plot(x0, STiy3, 'r.', label='插值点')
        # ax2.plot(x, SM.loc[i], label='原数值')
        # ax2.plot(x0, SMiy3, 'b.', label='插值点')

    SM_y = np.around(SM_y, decimals=4)
    ST_y = np.around(ST_y, decimals=4)
    # 为df插入两列
    dfSM_y = pd.DataFrame(SM_y)
    dfST_y = pd.DataFrame(ST_y)
    df['土壤温度(℃)-70'], df['土壤温度(℃)-90'] = dfST_y.iloc[:, 0], dfST_y.iloc[:, 1]
    df['水分含量(%)-70'], df['水分含量(%)-90'] = dfSM_y.iloc[:, 0], dfSM_y.iloc[:, 1]
    df.to_csv(path_or_buf=savepath, encoding='utf-8_sig', index=False)
    return


# 最大最小归一化
def scalerVariable(x, y, m, n):
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    X = scaler_x.fit_transform(x)
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    Y = scaler_y.fit_transform(y.reshape(x.shape[0], -1))
    # 注意为常量的bulk density和土壤组成 不进行缩放  24：63列
    X[:, m:n] = x[:, m:n]
    # 保存所有数各个列的最大最小值，用于测试集滚动预测
    # df = pd.concat([pd.DataFrame(x).max(),pd.DataFrame(x).min(),pd.DataFrame(y).max(),pd.DataFrame(y).min()],axis=1)
    # df.to_csv('./data/min_max.csv',encoding='utf-8_sig', index=False,header=None)
    return [X, Y], scaler_y


#  标准化 standardScaler()
def standardScalerVariable(x, y, m, n):
    scaler_x = StandardScaler()
    X = scaler_x.fit_transform(x)
    scaler_y = StandardScaler()
    Y = scaler_y.fit_transform(y.reshape(x.shape[0], -1))
    # 注意为常量的bulk density和土壤组成 不进行缩放
    # X[:, m:n] = x[:, m:n]
    return [X, Y], scaler_y
# 归一化
def NormalizerVariable(x, y, m, n):
    X = Normalizer().fit_transform(x)
    scaler_y = Normalizer()
    Y = scaler_y.fit_transform(y.reshape(x.shape[0], -1))
    # 注意为常量的bulk density和土壤组成 不进行缩放
    X[:, m:n] = x[:, m:n]
    return [X, Y], scaler_y


# 提取出待滚动预测的数据-待提交数据
def getTestdata(dataset, point):
    data = np.array(dataset)[561:, 6:100]
    if (point == 5):
        data = np.array(dataset)[346:, 6:100]
    testx, testy = sliding_window(data, 1, 0, 0)
    return [testx, testy]


if __name__ == '__main__':
    """
           整理拼接全部数据 气象+墒情
    """
    # 先处理预测点3的数据
    # interpolate('./data/rawData_csv/SMST3.csv','./data/rawData_csv/SMST3_new.csv')

    # pathdir1 = ['./data/rawData_csv/Meteorology1.csv', './data/rawData_csv/SMST1.csv',
    #             './data_addfeatures1/checkData/Meteorology1.csv', './data_addfeatures1/checkData/SMST1.csv', './data_addfeatures1/inputData2/variable1.csv']
    # pathdir2 = ['./data/rawData_csv/Meteorology2.csv', './data/rawData_csv/SMST2.csv',
    #             './data_addfeatures1/checkData/Meteorology2.csv', './data_addfeatures1/checkData/SMST2.csv', './data_addfeatures1/inputData2/variable2.csv']
    # pathdir3 = ['./data/rawData_csv/Meteorology3.csv', './data/rawData_csv/SMST3_new.csv',
    #             './data_addfeatures1/checkData/Meteorology3.csv', './data_addfeatures1/checkData/SMST3.csv', './data_addfeatures1/inputData2/variable3.csv']
    # pathdir4 = ['./data/rawData_csv/Meteorology4.csv', './data/rawData_csv/SMST4.csv',
    #             './data_addfeatures1/checkData/Meteorology4.csv', './data_addfeatures1/checkData/SMST4.csv', './data_addfeatures1/inputData2/variable4.csv']
    # pathdir5 = ['./data/rawData_csv/Meteorology5.csv', './data/rawData_csv/SMST5.csv',
    #             './data_addfeatures1/checkData/Meteorology5.csv', './data_addfeatures1/checkData/SMST5.csv', './data_addfeatures1/inputData2/variable5.csv']
    # pathdir6=['./data/rawData_csv/Meteorology6.csv','./data/rawData_csv/SMST6.csv',
    #           './data_addfeatures1/checkData/Meteorology6.csv','./data_addfeatures1/checkData/SMST6.csv','./data_addfeatures1/inputData2/variable6.csv']
    # # 读取土壤容重数据
    # bddf = pd.read_csv("./data/rawData_csv/bulk_density.csv",header=0)
    # # 读取土壤组成数据
    # soilComp = pd.read_csv('./data/rawData_csv/soilComponent.csv',header=0)
    # readrawData(pathdir1, bddf.loc[0],soilComp.loc[0])
    # readrawData(pathdir2, bddf.loc[1],soilComp.loc[1])
    # readrawData(pathdir3, bddf.loc[2],soilComp.loc[2])
    # readrawData(pathdir4, bddf.loc[3],soilComp.loc[3])
    # readrawData(pathdir5, bddf.loc[4],soilComp.loc[4]) # 注意 SMST5.csv的列名 多带了 cm 单位，为统一，需要手动删去(直接改rawdata里的列名)
    # readrawData(pathdir6,bddf.loc[5],soilComp.loc[5])

    # 读取前，处理一下dataset5的无用数据

    # 读取6个预测点的变量数据  read_csv 不需要关闭
    dataset1 = pd.read_csv("./data_addfeatures1/inputData2/variable1.csv", header=0)
    dataset2 = pd.read_csv("./data_addfeatures1/inputData2/variable2.csv", header=0)
    dataset3 = pd.read_csv("./data_addfeatures1/inputData2/variable3.csv", header=0)
    dataset4 = pd.read_csv("./data_addfeatures1/inputData2/variable4.csv", header=0)
    dataset5 = pd.read_csv("./data_addfeatures1/inputData2/variable5.csv", header=0)
    dataset6 = pd.read_csv("./data_addfeatures1/inputData2/variable6.csv", header=0)

    sliding_window_width = 1
    input_sequence_start = 0

    # 对每个预测点划分测试集和训练集
    train1, test1 = split_dataset(dataset1)
    train2, test2 = split_dataset(dataset2)
    train3, test3 = split_dataset(dataset3)
    train4, test4 = split_dataset(dataset4)
    train5, test5 = split_dataset_P5(dataset5)
    train6, test6 = split_dataset(dataset6)

    # --------------------------------------保存需要提交的测试数据-------------------------------------------
    # Testx1, Testy1 = getTestdata(dataset1, 1)
    # Testx2, Testy2 = getTestdata(dataset2, 2)
    # Testx3, Testy3 = getTestdata(dataset3, 3)
    # Testx4, Testy4 = getTestdata(dataset4, 4)
    # Testx5, Testy5 = getTestdata(dataset5, 5)
    # Testx6, Testy6 = getTestdata(dataset6, 6)
    # pd.DataFrame(Testx1).to_csv('./data/Test_x1.csv',encoding='utf-8_sig', index=False,header=None)
    # pd.DataFrame(Testy1).to_csv('./data/Test_y1.csv',encoding='utf-8_sig', index=False,header=None)
    # pd.DataFrame(Testx2).to_csv('./data/Test_x2.csv', encoding='utf-8_sig', index=False, header=None)
    # pd.DataFrame(Testy2).to_csv('./data/Test_y2.csv', encoding='utf-8_sig', index=False, header=None)
    # pd.DataFrame(Testx3).to_csv('./data/Test_x3.csv', encoding='utf-8_sig', index=False, header=None)
    # pd.DataFrame(Testy3).to_csv('./data/Test_y3.csv', encoding='utf-8_sig', index=False, header=None)
    # pd.DataFrame(Testx4).to_csv('./data/Test_x4.csv', encoding='utf-8_sig', index=False, header=None)
    # pd.DataFrame(Testy4).to_csv('./data/Test_y4.csv', encoding='utf-8_sig', index=False, header=None)
    # pd.DataFrame(Testx5).to_csv('./data/Test_x5.csv', encoding='utf-8_sig', index=False, header=None)
    # pd.DataFrame(Testy5).to_csv('./data/Test_y5.csv', encoding='utf-8_sig', index=False, header=None)
    # pd.DataFrame(Testx6).to_csv('./data/Test_x6.csv', encoding='utf-8_sig', index=False, header=None)
    # pd.DataFrame(Testy6).to_csv('./data/Test_y6.csv', encoding='utf-8_sig', index=False, header=None)

    # --------------------------------------------------------------------------------------------------------

    # 划分X和Y 不需要滑动窗口，一行就是一条数据
    train_x1, train_y1 = sliding_window(train1, sliding_window_width, in_start=input_sequence_start)
    train_x2, train_y2 = sliding_window(train2, sliding_window_width, in_start=input_sequence_start)
    train_x3, train_y3 = sliding_window(train3, sliding_window_width, in_start=input_sequence_start)
    train_x4, train_y4 = sliding_window(train4, sliding_window_width, in_start=input_sequence_start)
    train_x5, train_y5 = sliding_window(train5, sliding_window_width, in_start=input_sequence_start)
    train_x6, train_y6 = sliding_window(train6, sliding_window_width, in_start=input_sequence_start)

    test_x1, test_y1 = sliding_window(test1, sliding_window_width, in_start=input_sequence_start)
    test_x2, test_y2 = sliding_window(test2, sliding_window_width, in_start=input_sequence_start)
    test_x3, test_y3 = sliding_window(test3, sliding_window_width, in_start=input_sequence_start)
    test_x4, test_y4 = sliding_window(test4, sliding_window_width, in_start=input_sequence_start)
    test_x5, test_y5 = sliding_window(test5, sliding_window_width, in_start=input_sequence_start)
    test_x6, test_y6 = sliding_window(test6, sliding_window_width, in_start=input_sequence_start)

    # 拼接测试集，保存为csv
    test_x = np.concatenate((test_x1, test_x2, test_x3, test_x4, test_x5, test_x6), axis=0)
    test_y = np.concatenate((test_y1, test_y2, test_y3, test_y4, test_y5, test_y6), axis=0)
    test_x = pd.DataFrame(test_x)
    test_x.to_csv('./data_addfeatures1/test_x.csv',encoding='utf-8_sig', index=False,header=None)
    test_y = pd.DataFrame(test_y)
    test_y.to_csv('./data_addfeatures1/test_y.csv',encoding='utf-8_sig', index=False,header=None)

    # 拼接各个预测点的数据 多维数组拼接 x=> [1965，84]  y=>[1965，10]
    # 加入测试数据，全部预测
    train_x = np.concatenate((train_x1, train_x2, train_x3, train_x4, train_x5, train_x6), axis=0)
    train_y = np.concatenate((train_y1, train_y2, train_y3, train_y4, train_y5, train_y6), axis=0)

    # ---------------------------------------------10层一起预测的输入数据，用于lstm------------------------------------------
    # train, scaler_trainy  = dataForAllLayers(train_x,train_y)
    # ----------------------------------------------------------------------------------------------------------

    # 单层预测的输入数据
    train = [train_x, train_y]
    train_l1, scaler_trainy1 = dataForEachLayer(train, 1)
    train_l2, scaler_trainy2 = dataForEachLayer(train, 2)
    train_l3, scaler_trainy3 = dataForEachLayer(train, 3)
    train_l4, scaler_trainy4 = dataForEachLayer(train, 4)
    train_l5, scaler_trainy5 = dataForEachLayer(train, 5)
    train_l6, scaler_trainy6 = dataForEachLayer(train, 6)
    train_l7, scaler_trainy7 = dataForEachLayer(train, 7)
    train_l8, scaler_trainy8 = dataForEachLayer(train, 8)
    train_l9, scaler_trainy9 = dataForEachLayer(train, 9)
    train_l10, scaler_trainy10 = dataForEachLayer(train, 10)

    epochs_num = 1000
    batch_size_set = 50
    verbose_set = 2  # 显示每个epoch的记录
    # lightgbm 预测
    lightgbm_model(train_l1, './save_model1/lgb_model/lgb_level1.pkl',level=1)
    lightgbm_model(train_l2, './save_model1/lgb_model/lgb_level2.pkl',level=2)
    lightgbm_model(train_l3, './save_model1/lgb_model/lgb_level3.pkl',level=3)
    lightgbm_model(train_l4, './save_model1/lgb_model/lgb_level4.pkl',level=4)
    lightgbm_model(train_l5, './save_model1/lgb_model/lgb_level5.pkl',level=5)
    lightgbm_model(train_l6, './save_model1/lgb_model/lgb_level6.pkl',level=6)
    lightgbm_model(train_l7, './save_model1/lgb_model/lgb_level7.pkl',level=7)
    lightgbm_model(train_l8, './save_model1/lgb_model/lgb_level8.pkl',level=8)
    lightgbm_model(train_l9, './save_model1/lgb_model/lgb_level9.pkl',level=9)
    lightgbm_model(train_l10, './save_model1/lgb_model/lgb_level10.pkl',level=10)

    # XGBoost预测
    # xgboost_model(train_l1, './save_model/xgb_model_all/xgb_level1.pkl',level=1)
    # xgboost_model(train_l2, './save_model/xgb_model_all/xgb_level2.pkl',level=2)
    # xgboost_model(train_l3, './save_model/xgb_model_all/xgb_level3.pkl',level=3)
    # xgboost_model(train_l4, './save_model/xgb_model_all/xgb_level4.pkl',level=4)
    # xgboost_model(train_l5, './save_model/xgb_model_all/xgb_level5.pkl',level=5)
    # xgboost_model(train_l6, './save_model/xgb_model_all/xgb_level6.pkl',level=6)
    # xgboost_model(train_l7, './save_model/xgb_model_all/xgb_level7.pkl',level=7)
    # xgboost_model(train_l8, './save_model/xgb_model_all/xgb_level8.pkl',level=8)
    # xgboost_model(train_l9, './save_model/xgb_model_all/xgb_level9.pkl',level=9)
    # xgboost_model(train_l10, './save_model/xgb_model_all/xgb_level10.pkl',level=10)

    # SVR 预测
    # svr_model(train_l1,'./save_model/svr_model/svr_level1.pkl')
    # svr_model(train_l2, './save_model/svr_model/svr_level2.pkl')
    # svr_model(train_l3, './save_model/svr_model/svr_level3.pkl')
    # svr_model(train_l4, './save_model/svr_model/svr_level4.pkl')
    # svr_model(train_l5, './save_model/svr_model/svr_level5.pkl')
    # svr_model(train_l6, './save_model/svr_model/svr_level6.pkl')
    # svr_model(train_l7, './save_model/svr_model/svr_level7.pkl')
    # svr_model(train_l8, './save_model/svr_model/svr_level8.pkl')
    # svr_model(train_l9, './save_model/svr_model/svr_level9.pkl')
    # svr_model(train_l10, './save_model/svr_model/svr_level10.pkl')

    # RF 预测
    # rf_model(train_l1, './save_model/rf_model_all/rf_level1.pkl', level=1)
    # rf_model(train_l2, './save_model/rf_model_all/rf_level2.pkl', level=2)
    # rf_model(train_l3, './save_model/rf_model_all/rf_level3.pkl', level=3)
    # rf_model(train_l4, './save_model/rf_model_all/rf_level4.pkl', level=4)
    # rf_model(train_l5, './save_model/rf_model_all/rf_level5.pkl', level=5)
    # rf_model(train_l6, './save_model/rf_model_all/rf_level6.pkl', level=6)
    # rf_model(train_l7, './save_model/rf_model_all/rf_level7.pkl', level=7)
    # rf_model(train_l8, './save_model/rf_model_all/rf_level8.pkl', level=8)
    # rf_model(train_l9, './save_model/rf_model_all/rf_level9.pkl', level=9)
    # rf_model(train_l10, './save_model/rf_model_all/rf_level10.pkl', level=10)

    print("End!!")














