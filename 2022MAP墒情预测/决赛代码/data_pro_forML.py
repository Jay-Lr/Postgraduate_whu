"""
    
    ML系列模型，15天一起预测/单天预测
    
    interpolate:缺失变量插值
    readrawdata:提取 t-1,t-2墒情，t-(t+14)墒情，t土壤温度，t-(t+14)气象平均，月、年中天，
                每个变量保存为一个csv文件
    concatVariable: 读取各个变量并拼接
    split_dataset: 每个预测点划分训练，测试集
    
    dataForEachDay:获取15天同时预测的数据（设置day=0生成15天或单天的数据集）
    dataForEachLayer: 获取单层的输入数据
    
    模型：
        svr_model
        rf_model
        lightgbm_model
        xgboost_model

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
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
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

def plot_features(dataset,picname):
    # 创建一个包含n个子图的图像，每个子图对应一个变量
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
    # plt.show()
    plt.savefig(dataset.columns[1]+picname+'.jpg')



# 支持向量机模型
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


# 随机森林模型
def rf_model(train, path, level):
    [train_x, train_y] = train
    train_y = np.ravel(train_y)  # 改成(shapes,)
    if (level == 1):
        param_grid = {
            'criterion': ['mse'],
            'n_estimators': [55],  # 决策树个数
            'max_depth': [23],  # 深度：这里是森林中每棵决策树的深度
            # 'max_features': np.arange(8,30,2), # 每棵决策树使用的变量占比
            # 'min_samples_split': [4, 8, 12, 16]  # 叶子的最小拆分样本量
        }
    elif (level == 2):
        param_grid = {
            'criterion': ['mse'],
            'n_estimators': [55],  # 决策树个数
            'max_depth': [20],  # 深度：这里是森林中每棵决策树的深度
        }
    elif (level == 3):
        param_grid = {
            'criterion': ['mse'],
            'n_estimators': [55],  # 决策树个数
            'max_depth': [20],  # 深度：这里是森林中每棵决策树的深度
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
    """
        划分测试集和训练集：
        9,54;6,55;8,41;8,45;9,35;8,38;7,48;9,52;8,45;8,45;
        全部数据测试：
        9,50;7,56;8,58;8,45;9,36;7,35;6,36;8,44;8,45;7,46;

    """
    rfr = RandomForestRegressor()
    rfr_cv = GridSearchCV(estimator=rfr, param_grid=param_grid,
                          scoring='neg_mean_squared_error', cv=5)  # neg_mean_squared_error
    rfr_cv.fit(train_x, train_y)

    print(rfr_cv.best_params_)  # 查看梯度优化给出的最佳参数，如果参数落在了给定范围（param_grid）的边界上，需要调整参数的范围
    print(rfr_cv.best_score_)  # 0.940034

    # importances = list(rfr_cv.feature_importances_)
    # print(importances)
    joblib.dump(rfr_cv, path)
    return


# lightgbm
def lightgbm_model(train, path, level):
    # 训练集
    [train_x, train_y] = train
    train_y = np.ravel(train_y)
    """
       
    """
    # lightgbm模型参数设置
    cv_params = {
        # 'n_estimators':np.arange(2000,4000,100),
        # 'max_depth': [9,12,15,19,22,25],
        # 'num_leaves': np.arange(20, 30, 2),
        # 'subsample': [0.6, 0.7, 0.8, 0.9, 1],
        # 'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1],
        # 'bagging_freq':[1,2],#  4, 5, 6, 8
        # 'reg_alpha': np.logspace(-2, 1, num=6, base=2.0),
        # 'reg_lambda': np.logspace(-2, 1, num=6, base=2.0)
    }
    """
        首先确定较大的学习率，0.1
        调整 n_estimators，max_depth，num_leaves，colsample_bytree，subsample，bagging_freq
        确定L1L2正则reg_alpha和reg_lambda
        降低调整学习率,如0.03,0.01,0.00*,…… 
         'n_estimators':np.arange(200,550,10),
        'learning_rate':np.logspace(-2, -1, num=15),
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'num_leaves':np.arange(2,16,1),
         'subsample': [0.6, 0.7, 0.8, 0.9, 1],
          'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1], 
           'bagging_freq': [2, 4, 5, 6, 8]
        'reg_alpha': np.logspace(-2, 1, num=6),
          'reg_lambda': np.logspace(-2, 1, num=6),       

    """
    if (level == 1):
        params = {
            'boosting_type': 'gbdt',  # 使用的树的模型：梯度提升决策树（GBDT）
            'objective': 'regression',  # 目标函数 使用L2正则项的回归模型（默认值)
            'metric': 'mse',  # 评估函数
            'n_estimators': 4000,  # 分类器数量 5000
            'learning_rate': 0.1,  # 学习速率  [0.01, 0.015, 0.025, 0.05, 0.1]
            'max_depth': 12,  # 树的最大深度 [3, 5, 6, 7, 9, 12, 15, 17, 25]
            'num_leaves': 30,  # 一般设置为(0, 2^max_depth - 1]的一个数值
            'colsample_bytree': 0.9,  # 建树的特征选择比例 [0.6, 0.7, 0.8, 0.9, 1]
            'subsample': 0.9,  # 建树的样本采样比例 [0.6, 0.7, 0.8, 0.9, 1]
            'bagging_freq': 3,  # k 意味着每 k 次迭代执行bagging
            'reg_alpha': 0.25,
            'reg_lambda': 0.574,
        }
    elif (level == 2):
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'mse',
            'n_estimators': 4000,
            'learning_rate': 0.1,
            'max_depth': 12,
            'num_leaves': 32,
            'colsample_bytree': 0.9,
            'subsample': 0.8,
            'bagging_freq': 1,
            'reg_alpha': 0,
            'reg_lambda': 0.5,
        }
    elif (level == 3):
        params = {
            'boosting_type': 'gbdt',  # 使用的树的模型：梯度提升决策树（GBDT）
            'objective': 'regression',  # 目标函数 使用L2正则项的回归模型（默认值)
            'metric': 'mse',  # 评估函数
            'n_estimators': 3000,  # 分类器数量
            'learning_rate': 0.1,  # 学习速率  [0.01, 0.015, 0.025, 0.05, 0.1]
            'max_depth': 15,  # 树的最大深度 [3, 5, 6, 7, 9, 12, 15, 17, 25]
            'num_leaves': 34,  # 一般设置为(0, 2^max_depth - 1]的一个数值
            'colsample_bytree': 1,  # 建树的特征选择比例 [0.6, 0.7, 0.8, 0.9, 1]
            'subsample': 0.8,  # 建树的样本采样比例 [0.6, 0.7, 0.8, 0.9, 1]
            'bagging_freq': 3,  # k 意味着每 k 次迭代执行bagging  [2, 4, 5, 6, 8]
            'reg_alpha': 0.01,
            'reg_lambda': 0.57,
        }
    elif (level == 4):
        params = {
            'boosting_type': 'gbdt',  # 使用的树的模型：梯度提升决策树（GBDT）
            'objective': 'regression',  # 目标函数 使用L2正则项的回归模型（默认值)
            'metric': 'mse',  # 评估函数
            'n_estimators': 3000,  # 分类器数量
            'learning_rate': 0.1,  # 学习速率  [0.01, 0.015, 0.025, 0.05, 0.1]
            'max_depth': 12,  # 树的最大深度 [3, 5, 6, 7, 9, 12, 15, 17, 25]
            'num_leaves': 28,  # 一般设置为(0, 2^max_depth - 1]的一个数值
            'colsample_bytree': 0.8,  # 建树的特征选择比例 [0.6, 0.7, 0.8, 0.9, 1]
            'subsample': 0.8,  # 建树的样本采样比例 [0.6, 0.7, 0.8, 0.9, 1]
            'bagging_freq': 1,  # k 意味着每 k 次迭代执行bagging  [2, 4, 5, 6, 8]
            'reg_alpha': 0.25,
            'reg_lambda': 2,
        }
    elif (level == 5):
        params = {
            'boosting_type': 'gbdt',  # 使用的树的模型：梯度提升决策树（GBDT）
            'objective': 'regression',  # 目标函数 使用L2正则项的回归模型（默认值)
            'metric': 'mse',  # 评估函数
            'n_estimators': 3000,  # 分类器数量
            'learning_rate': 0.1,  # 学习速率  [0.01, 0.015, 0.025, 0.05, 0.1]
            'max_depth': 15,  # 树的最大深度 [3, 5, 6, 7, 9, 12, 15, 17, 25]
            'num_leaves': 30,  # 一般设置为(0, 2^max_depth - 1]的一个数值
            'colsample_bytree': 1,  # 建树的特征选择比例 [0.6, 0.7, 0.8, 0.9, 1]
            'subsample': 0.8,  # 建树的样本采样比例 [0.6, 0.7, 0.8, 0.9, 1]
            'bagging_freq': 2,  # k 意味着每 k 次迭代执行bagging  [2, 4, 5, 6, 8]
            'reg_alpha': 0.03,
            'reg_lambda': 0.25,
        }
    elif (level == 6):
        params = {
            'boosting_type': 'gbdt',  # 使用的树的模型：梯度提升决策树（GBDT）
            'objective': 'regression',  # 目标函数 使用L2正则项的回归模型（默认值)
            'metric': 'mse',  # 评估函数
            'n_estimators': 3000,  # 分类器数量
            'learning_rate': 0.1,  # 学习速率  [0.01, 0.015, 0.025, 0.05, 0.1]
            'max_depth': 9,  # 树的最大深度 [3, 5, 6, 7, 9, 12, 15, 17, 25]
            'num_leaves': 22,  # 一般设置为(0, 2^max_depth - 1]的一个数值
            'colsample_bytree': 1,  # 建树的特征选择比例 [0.6, 0.7, 0.8, 0.9, 1]
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
            'n_estimators': 3000,  # 分类器数量
            'learning_rate': 0.1,  # 学习速率  [0.01, 0.015, 0.025, 0.05, 0.1]
            'max_depth': 15,  # 树的最大深度 [3, 5, 6, 7, 9, 12, 15, 17, 25]
            'num_leaves': 38,  # 一般设置为(0, 2^max_depth - 1]的一个数值
            'colsample_bytree': 1,  # 建树的特征选择比例 [0.6, 0.7, 0.8, 0.9, 1]
            'subsample': 0.9,  # 建树的样本采样比例 [0.6, 0.7, 0.8, 0.9, 1]
            'bagging_freq': 3,  # k 意味着每 k 次迭代执行bagging  [2, 4, 5, 6, 8]
            'reg_alpha': 0.25,
            'reg_lambda': 0.38,
        }
    elif (level == 8):
        params = {
            'boosting_type': 'gbdt',  # 使用的树的模型：梯度提升决策树（GBDT）
            'objective': 'regression',  # 目标函数 使用L2正则项的回归模型（默认值)
            'metric': 'mse',  # 评估函数
            'n_estimators': 3000,  # 分类器数量
            'learning_rate': 0.1,  # 学习速率  [0.01, 0.015, 0.025, 0.05, 0.1]
            'max_depth': 10,  # 树的最大深度 [3, 5, 6, 7, 9, 12, 15, 17, 25]
            'num_leaves': 26,  # 一般设置为(0, 2^max_depth - 1]的一个数值
            'colsample_bytree': 1,  # 建树的特征选择比例 [0.6, 0.7, 0.8, 0.9, 1]
            'subsample': 0.7,  # 建树的样本采样比例 [0.6, 0.7, 0.8, 0.9, 1]
            'bagging_freq': 4,  # k 意味着每 k 次迭代执行bagging  [2, 4, 5, 6, 8]
            'reg_alpha': 0,
            'reg_lambda': 0.37,
        }
    elif (level == 9):
        params = {
            'boosting_type': 'gbdt',  # 使用的树的模型：梯度提升决策树（GBDT）
            'objective': 'regression',  # 目标函数 使用L2正则项的回归模型（默认值)
            'metric': 'mse',  # 评估函数
            'n_estimators': 3000,  # 分类器数量
            'learning_rate': 0.1,  # 学习速率  [0.01, 0.015, 0.025, 0.05, 0.1]
            'max_depth': 9,  # 树的最大深度 [3, 5, 6, 7, 9, 12, 15, 17, 25]
            'num_leaves': 26,  # 一般设置为(0, 2^max_depth - 1]的一个数值
            'colsample_bytree': 1,  # 建树的特征选择比例 [0.6, 0.7, 0.8, 0.9, 1]
            'subsample': 0.9,  # 建树的样本采样比例 [0.6, 0.7, 0.8, 0.9, 1]
            'bagging_freq': 1,  # k 意味着每 k 次迭代执行bagging  [2, 4, 5, 6, 8]
            'reg_alpha': 0,
            'reg_lambda': 0.31,
        }
    elif (level == 10):
        params = {
            'boosting_type': 'gbdt',  # 使用的树的模型：梯度提升决策树（GBDT）
            'objective': 'regression',  # 目标函数 使用L2正则项的回归模型（默认值)
            'metric': 'mse',  # 评估函数
            'n_estimators': 2000,  # 分类器数量
            'learning_rate': 0.1,  # 学习速率  [0.01, 0.015, 0.025, 0.05, 0.1]
            'max_depth': 12,  # 树的最大深度 [3, 5, 6, 7, 9, 12, 15, 17, 25]
            'num_leaves': 28,  # 一般设置为(0, 2^max_depth - 1]的一个数值
            'colsample_bytree': 0.9,  # 建树的特征选择比例 [0.6, 0.7, 0.8, 0.9, 1]
            'subsample': 0.9,  # 建树的样本采样比例 [0.6, 0.7, 0.8, 0.9, 1]
            'bagging_freq': 2,  # k 意味着每 k 次迭代执行bagging  [2, 4, 5, 6, 8]
            'reg_alpha': 0,
            'reg_lambda': 0,
        }

    model = lgb.LGBMRegressor(**params)

    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=5,
                                 verbose=1, n_jobs=4)
    optimized_GBM.fit(train_x, train_y, verbose=False)
    # model.fit(train_x, train_y, verbose=False)

    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
    joblib.dump(optimized_GBM, path)
    # df1 = pd.DataFrame()
    # df1['feature name']=['month','yearofday','bulk density','sand','silt','clay','ST0','STN','SM_t-2','SM_t-1','相对湿度(%)','大气压力(hPa)',
    #             '风速(m/s)','风向(°)','雨量(mm)','当前太阳辐射强度(W/m²)','累计太阳辐射量(MJ/m²)','空气温度(℃)','day']
    # df1['importance']= model.feature_importances_
    # df1.sort_values(by='importance', inplace=True, ascending=False)
    # df1.plot.barh(x='feature name',y='importance')
    # lgb.plot_importance(model, max_num_features=10)
    # plt.title("Featurertances")
    # plt.show(block=True)

    return


# Xgboost
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
            'learning_rate': 0.1, 'n_estimators': 720,
            'max_depth': 10, 'min_child_weight': 3, 'seed': 0,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'gamma': 0, 'reg_alpha': 0.25, 'reg_lambda': 0
        }
    elif (level == 6):
        other_params = {
            'learning_rate': 0.1, 'n_estimators': 720,
            'max_depth': 10, 'min_child_weight': 5, 'seed': 0,
            'subsample': 0.9, 'colsample_bytree': 1,
            'gamma': 0, 'reg_alpha': 0.25, 'reg_lambda': 0
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

    """

        训练数据
        'n_estimators':np.arange(700,750,5),
        'learning_rate':np.logspace(-2, -1, num=5),
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'min_child_weight': [1, 2, 3, 4, 5, 6],
         'gamma': np.logspace(-3, 0, num=6, base=2.0),
         'subsample': [0.6, 0.7, 0.8, 0.9, 1],
          'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1],
         'reg_alpha': np.logspace(-2, 1, num=6, base=2.0),
          'reg_lambda': np.logspace(-2, 1, num=6, base=2.0)      
    """
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
    joblib.dump(optimized_GBM, path)
    return



def gbdt_model(train,path,level):
    [train_x, train_y] = train
    train_y = np.ravel(train_y)
    cv_params = {
        'n_estimators': np.arange(100, 550, 10),
        # 'max_depth': [3, 5, 7, 9, 12,15],
        # 'num_leaves': np.arange(2, 16, 1),
    }
    other_params = {'n_estimators': 520, 'max_depth': 5, 'min_samples_split': 2,
              'learning_rate': 0.01, 'loss': 'ls'} # 损失函数：最小二乘
    # 估计器拟合训练数据
    model = GradientBoostingRegressor(**other_params)
    optimized_GBR = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=5,
                                 verbose=1, n_jobs=4)
    optimized_GBR.fit(train_x, train_y)


    print('参数的最佳取值：{0}'.format(optimized_GBR.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBR.best_score_))
    joblib.dump(optimized_GBR, path)
    return

# 划分训练，测试集
def split_dataset(data):
    '''
        该函数切分训练数据和测试数据  7:3
    '''
    data = np.array(data)
    # 设置随机数种子，保证每次生成的结果都是一样的
    np.random.seed(42)
    # permutation随机生成0-len(data)随机序列
    shuffled_indices = np.random.permutation(len(data))
    # test_ratio为测试集所占的半分比
    test_set_size = int(len(data) * 0.3)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    train, test = data[train_indices], data[test_indices]
    return np.array(train), np.array(test)


# 滑动窗口，整理输入，输出变量
def sliding_window(train, sw_width=1, n_out=1, in_start=0):
    # data = train.reshape((train.shape[0], train.shape[1]))  # 二维 [n,]
    X = train[:, 0:-150]
    # t时刻23：00土壤湿度为预测值
    Y = train[:, -150:]
    return np.array(X), np.array(Y)

def dataForEachDay(data,flag):
    """
    :param data:
    :param flag: 为T,15天的数据按行拼接在一起；为F,提取出每天的数据，得到X，Y为三维数组，[samples,features,days]
    :return:
    """
    [x, y] = data
    x1=np.concatenate([x[:,0:53],x[:,173:193]],axis=1)  # 土壤组成数据,温度数据,历史墒情
    X=[]
    Y=[]
    for i in range(15):
        x2 = x[:,(53+i*8):(53+(i+1)*8)]
        yy = y[:,i*10:(i+1)*10]
        xx =np.concatenate([x1,x2],axis=1)
        if(flag):
            # 为每天的数据增加一个天数变量
            xx=pd.DataFrame(xx)
            xx['day']=i+1
            xx = np.array(xx)
            #  15天的要一起计算，则把15天的数据拼在一起
            if(i==0):
                X=xx
                Y=yy
            else:
                X = np.append(X,xx,axis=0) # 沿行拼接在一起
                Y = np.append(Y, yy, axis=0)
        else:
            # 15天，每天为一维数据
            if(i==0):
                X=xx
                Y=yy
            elif(i==1):
                X = np.stack([X,xx],axis=-1)
                Y = np.stack([Y, yy],axis=-1)
            else:
                X = np.append(X,xx.reshape(xx.shape[0],-1,1),axis=2)
                Y = np.append(Y, yy.reshape(yy.shape[0], -1, 1), axis=2)  #【n,f,15】
    return [X,Y]


# 划分每层数据并归一化
def dataForEachLayer(data, level,day):
    """
        :param x: 数组x
        :param y: 数组y
        :param level: 土壤层数
        :param day: 天数，
            0：全部天数的数据一起训练；
            1-15，分别提取每天的数据；
        :return:
    """
    random.seed(5)
    [x, y] = data
    # day=0 全部天一起训练
    if(day!=0):
        # 每天单独训练
        x=x[:,:,day-1]
        y=y[:,:,day-1]
    index = [i for i in range(len(x))]
    random.shuffle(index)
    x = x[index]
    y = y[index]
    x_date = x[:,0:2]
    x_bd = x[:, 2 + level - 1]
    x_sand = x[:, 12 + level - 1]
    x_silt = x[:, 22 + level - 1]
    x_clay = x[:, 32 + level - 1]
    x_st= np.c_[x[:,42],x[:,43+level-1]]  # 地表温度和level层温度
    x_t_2 = x[:, 53 + level - 1]
    x_t_1 = x[:, 63 + level - 1]
    x_meteo = x[:, 73:]
    X = np.c_[x_date,x_bd, x_sand, x_silt, x_clay,x_st,x_t_2, x_t_1,  x_meteo]  # 全部天一起训练时，多出了一个 天数 变量
    Y = y[:, level - 1]
    # 先划分再归一化
    data, scaler_y = scalerVariable(X, Y, 0, 0)
    return data, scaler_y


# 处理原始数据
def readrawData(point,rootpath,pathdir, bulk_density, soilComp):
    """
        预测点1,2,3,4，分辨率为小时级，
        预测点 5,6，分辨率为分钟级，
        使用小时采样保证时间连续且插值
    """
    # 首行为列名，第一列时间列为索引列 dataframe
    df = pd.read_csv(pathdir[0], header=0)
    df['时间'] = pd.to_datetime(df['时间'])
    df.set_index("时间", inplace=True)
    # 小于0的数据赋值为0
    # df[df[["相对湿度(%)","大气压力(hPa)","风速(m/s)","风向(°)","雨量(mm)","当前太阳辐射强度(W/m²)","累计太阳辐射量(MJ/m²)"]] < 0] = 0
    # todo 限制雨量，风速数值
    df['雨量(mm)']=np.where(df['雨量(mm)']>40, 40, df['雨量(mm)'])
    df['风速(m/s)'] = np.where(df['风速(m/s)'] > 20, 20, df['风速(m/s)'])
    df1 = pd.read_csv(pathdir[1], header=0)
    df1['时间'] = pd.to_datetime(df1['时间'])
    df1.set_index("时间", inplace=True)
    df1["水分含量(%)-10"] = df1["水分含量(%)-10"].astype("float64")  #
    plot_features(df1, picname=str(point))

    # 检查时间序列是否连续(使用小时重采样实现)
    df0 = df.resample('H').mean().interpolate()
    df2 = df1.resample('H').mean().interpolate()
    # 保存时间连续的数据 ,确定列名顺序
    order=["空气温度(℃)","相对湿度(%)","大气压力(hPa)","风速(m/s)","风向(°)","雨量(mm)","当前太阳辐射强度(W/m²)","累计太阳辐射量(MJ/m²)"]
    df0=df0[order]
    df0.to_csv(path_or_buf=pathdir[2], index_label="时间", encoding='utf_8_sig')
    plot_features(df0,picname=str(point))
    order=["土壤温度(℃)-地表",'土壤温度(℃)-10', '土壤温度(℃)-20', '土壤温度(℃)-30', '土壤温度(℃)-40',
             '土壤温度(℃)-50', '土壤温度(℃)-60', '土壤温度(℃)-70', '土壤温度(℃)-80', '土壤温度(℃)-90', '土壤温度(℃)-100' ,'水分含量(%)-10', '水分含量(%)-20', '水分含量(%)-30', '水分含量(%)-40',
             '水分含量(%)-50', '水分含量(%)-60', '水分含量(%)-70', '水分含量(%)-80', '水分含量(%)-90', '水分含量(%)-100']
    df2 = df2[order]
    df2.to_csv(path_or_buf=pathdir[3], index_label="时间", encoding='utf_8_sig')
    plot_features(df2, picname=str(point))

    # 计算气象因子的日均值  按时间重采样,计算变量日均值，label=right是指 索引保留右侧的取值，[0:00,23:00]
    df_day = df0.resample('D', label='left').mean().reset_index()

    # 取出每天23:00的数据  x指 行索引，其中，表头为0
    df_23h = pd.read_csv(pathdir[3], header=0, skiprows=lambda x: x > 0 and x % 24 != 0)
    df_23h['时间'] = pd.to_datetime(df_23h['时间'])
    # 提取温度数据
    df_ST = df_23h[["时间","土壤温度(℃)-地表","土壤温度(℃)-10","土壤温度(℃)-20","土壤温度(℃)-30","土壤温度(℃)-40","土壤温度(℃)-50",
                    "土壤温度(℃)-60","土壤温度(℃)-70","土壤温度(℃)-80","土壤温度(℃)-90","土壤温度(℃)-100"]]
    df_ST.to_csv(os.path.join(rootpath,"ST_23h"+str(point)+".csv"),encoding='utf_8_sig',index=False)
    # 增加 月和年中日数据 ，并保存为csv
    df_23h['month'] = df_23h['时间'].dt.month
    df_23h['days'] = df_23h['时间'].dt.dayofyear
    df_23h[['时间','month','days']].to_csv(os.path.join(rootpath,"date"+str(point)+".csv"),encoding='utf_8_sig',index=False)

    # 仅保留需要数据
    df_23h = df_23h[["时间",'水分含量(%)-10', '水分含量(%)-20', '水分含量(%)-30', '水分含量(%)-40',
                     '水分含量(%)-50', '水分含量(%)-60', '水分含量(%)-70', '水分含量(%)-80', '水分含量(%)-90', '水分含量(%)-100']]
    # 取23点土壤湿度t-2,t-1的滞后
    df_23h_lay2 = df_23h.shift(1).add_suffix("_t-1")
    df_23h_lay1 = df_23h.shift(0).add_suffix("_t")
    df_23h_lay1=df_23h_lay1.drop(["时间_t"],axis=1)
    SM = pd.concat([df_23h_lay2, df_23h_lay1], axis=1)
    # 增加当天+15 天的 墒情数据
    # 对于最后几天，值为nan
    for i in range(1,16):
        df_23h_forwi = df_23h.shift(-i).add_suffix("_t+"+str(i)).iloc[:,1:]
        SM = pd.concat([SM, df_23h_forwi],axis=1)
        for j in range(0,i+1):
            # 计算t-(t+i)天内的气象因素：
            df0_forwj = df_day.shift(-j).add_suffix("_t+" + str(j)).iloc[:,1:]
            if j ==0:
                df0_forw_ave = df_day.shift(0).iloc[:,1:]  #　舍弃 时间列
            else:
                df0_forw_ave.iloc[:,:]=np.array(df0_forw_ave)+np.array(df0_forwj)
        df0_forw_ave1=(df0_forw_ave/(i+1)).add_suffix("_t+" + str(i)+"ave")
        # 雨量 计算累计值
        df0_forw_ave1.iloc[:,4]= df0_forw_ave.iloc[:, 4]
        # 拼接 未来m天累计气象因子
        if(i==1): meteo = pd.concat([df_day.iloc[:,0],df0_forw_ave1], axis=1) # 第一列为时间列
        else:
            meteo=pd.concat([meteo,df0_forw_ave1], axis=1)
    # 保存t-2,t-1,t-(t+14)的SM数据
    SM.to_csv(os.path.join(rootpath,"SM_15d"+str(point)+".csv"),encoding='utf-8_sig', index=False)
    # 保存 t-(t+14)的气象均值数据
    meteo.to_csv(os.path.join(rootpath,"meto_ave"+str(point)+".csv"), encoding='utf-8_sig', index=False)


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
        # R_square.append([r2_score(ST.loc[i],spi.splev(x, STipo3).tolist()), r2_score(SM.loc[i], spi.splev(x0, SMipo3).tolist())])
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
    # 保存所有数各个列的最大最小值，用于测试集
    # if os.path.exists('data/submitData/min_max_st_all.csv'):
    #     data = pd.read_csv('data/submitData/min_max_st_all.csv', header=None)
    #     df = pd.concat([pd.DataFrame(x).max(), pd.DataFrame(x).min(), pd.DataFrame(y).max(), pd.DataFrame(y).min()],axis=1) # 左右拼接
    #     data=data.append(df)  # 添加新行
    #     data.to_csv('./data/min_max_st_all.csv', encoding='utf-8_sig', index=False, header=None) # 每行为每一层的 最大最小值
    # else:
    #     pd.concat([pd.DataFrame(x).max(), pd.DataFrame(x).min(), pd.DataFrame(y).max(), pd.DataFrame(y).min()],axis=1).to_csv(
    #         'data/submitData/min_max_st_all.csv', encoding='utf-8_sig', index=False, header=None)
    return [X, Y], scaler_y

# 读取各个变量并拼接
def concatVariable(date_path,meto_ave_path,SM_15d_path,ST_23h_path,bulk_density,soilComponent,savepath):
    # 各个csv有标题行，首列为时间列
    date = pd.read_csv(date_path, header=0)
    meto_ave = pd.read_csv(meto_ave_path, header=0)
    ST_23h = pd.read_csv(ST_23h_path, header=0)
    SM_15d = pd.read_csv(SM_15d_path, header=0)
    date['bd10'], date['bd20'], date['bd30'], date['bd40'], date['bd50'], date[
        'bd60'], date['bd70'], date['bd80'], date['bd90'], date['bd100'] =bulk_density
    date['sand10'], date['sand20'], date['sand30'], date['sand40'], date['sand50'], \
    date['sand60'], date['sand70'], date['sand80'], date['sand90'], date['sand100'], \
    date['silt10'], date['silt20'], date['silt30'], date['silt40'], date['silt50'], \
    date['silt60'], date['silt70'], date['silt80'], date['silt90'], date['silt100'], \
    date['clay10'], date['clay20'], date['clay30'], date['clay40'], date['clay50'], \
    date['clay60'], date['clay70'], date['clay80'], date['clay90'], date['clay100'] = soilComponent
    variable = pd.concat([date,ST_23h.iloc[:,1:],meto_ave.iloc[:,1:],SM_15d.iloc[:,1:]],axis=1)
    variable.to_csv(savepath, encoding='utf_8_sig',index=False)
    return


# 提取出待滚动预测的数据
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
    # interpolate('./data/rawData/STSM3.csv','./data/rawData/STSM3_new.csv')
    rootpath = "./data/inputData"
    pathdir1 = ['./data/rawData/Meteorology1.csv', './data/rawData/STSM1.csv',
                './data/checkData/Meteorology1.csv', './data/checkData/STSM1.csv', './data/inputData/variable1.csv']
    pathdir2 = ['./data/rawData/Meteorology2.csv', './data/rawData/STSM2.csv',
                './data/checkData/Meteorology2.csv', './data/checkData/STSM2.csv', './data/inputData/variable2.csv']
    pathdir3 = ['./data/rawData/Meteorology3.csv', './data/rawData/STSM3_new.csv',
                './data/checkData/Meteorology3.csv', './data/checkData/STSM3.csv', './data/inputData/variable3.csv']
    pathdir4 = ['./data/rawData/Meteorology4_new.csv', './data/rawData/STSM4_new.csv',
                './data/checkData/Meteorology4.csv', './data/checkData/STSM4.csv', './data/inputData/variable4.csv']
    pathdir5 = ['./data/rawData/Meteorology5.csv', './data/rawData/STSM5.csv',
                  './data/checkData/Meteorology5.csv', './data/checkData/STSM5.csv','./data/inputData/variable5.csv']
    pathdir5_1 = ['./data/rawData/Meteorology5_new1.csv', './data/rawData/STSM5_new1.csv',
                './data/checkData/Meteorology5_1.csv', './data/checkData/STSM5_1.csv', './data/inputData/variable5_1.csv']
    pathdir5_2 = ['./data/rawData/Meteorology5_new2.csv', './data/rawData/STSM5_new2.csv',
                './data/checkData/Meteorology5_2.csv', './data/checkData/STSM5_2.csv', './data/inputData/variable5_2.csv']
    pathdir5_3 = ['./data/rawData/Meteorology5_new3.csv', './data/rawData/STSM5_new3.csv',
                './data/checkData/Meteorology5_3.csv', './data/checkData/STSM5_3.csv', './data/inputData/variable5_3.csv']
    pathdir6_1=['./data/rawData/Meteorology6_new1.csv','./data/rawData/STSM6_new1.csv',
              './data/checkData/Meteorology6_1.csv','./data/checkData/STSM6_1.csv','./data/inputData/variable6_1.csv']
    pathdir6_2=['./data/rawData/Meteorology6_new2.csv','./data/rawData/STSM6_new2.csv',
              './data/checkData/Meteorology6_2.csv','./data/checkData/STSM6_2.csv','./data/inputData/variable6_2.csv']
    # 读取土壤容重数据
    bddf = pd.read_csv("./data/rawData/bulk_density.csv",header=0)
    # 读取土壤组成数据
    soilComp = pd.read_csv('./data/rawData/soilComponent.csv',header=0)
    # readrawData(1,rootpath,pathdir1, bddf.loc[0],soilComp.loc[0])
    # readrawData(2,rootpath,pathdir2, bddf.loc[1],soilComp.loc[1])
    # readrawData(3,rootpath,pathdir3, bddf.loc[2],soilComp.loc[2])
    # readrawData(4,rootpath,pathdir4, bddf.loc[3],soilComp.loc[3])
    readrawData(5, rootpath, pathdir5, bddf.loc[4], soilComp.loc[4])
    readrawData(5_1,rootpath,pathdir5_1, bddf.loc[4],soilComp.loc[4])
    readrawData(5_2, rootpath, pathdir5_2, bddf.loc[4], soilComp.loc[4])
    readrawData(5_3, rootpath, pathdir5_3, bddf.loc[4], soilComp.loc[4])
    readrawData(6_1,rootpath,pathdir6_1,bddf.loc[5],soilComp.loc[5])
    readrawData(6_2, rootpath, pathdir6_2, bddf.loc[5], soilComp.loc[5])
    date_path = ["./data/inputData/date1.csv","./data/inputData/date2.csv","./data/inputData/date3.csv",
    "./data/inputData/date4.csv","./data/inputData/date51.csv","./data/inputData/date52.csv","./data/inputData/date53.csv","./data/inputData/date61.csv","./data/inputData/date62.csv"]
    meto_ave_path = ["./data/inputData/meto_ave1.csv","./data/inputData/meto_ave2.csv","./data/inputData/meto_ave3.csv",
    "./data/inputData/meto_ave4.csv","./data/inputData/meto_ave51.csv","./data/inputData/meto_ave52.csv","./data/inputData/meto_ave53.csv","./data/inputData/meto_ave61.csv","./data/inputData/meto_ave62.csv"]
    SM_15d_path = ["data\inputData\SM_15d1.csv","data\inputData\SM_15d2.csv","data\inputData\SM_15d3.csv",
                   "data\inputData\SM_15d4.csv","data\inputData\SM_15d51.csv","data\inputData\SM_15d52.csv","data\inputData\SM_15d53.csv","data\inputData\SM_15d61.csv","data\inputData\SM_15d62.csv",]
    ST_23h_path = ["data\inputData\ST_23h1.csv","data\inputData\ST_23h2.csv","data\inputData\ST_23h3.csv",
                   "data\inputData\ST_23h4.csv","data\inputData\ST_23h51.csv","data\inputData\ST_23h52.csv","data\inputData\ST_23h53.csv","data\inputData\ST_23h61.csv","data\inputData\ST_23h62.csv"]
    savepath=["./data/inputData/variable_st1.csv","./data/inputData/variable_st2.csv","./data/inputData/variable_st3.csv",
              "./data/inputData/variable_st4.csv","./data/inputData/variable_st51.csv","./data/inputData/variable_st52.csv","./data/inputData/variable_st53.csv","./data/inputData/variable_st61.csv","./data/inputData/variable_st62.csv"]
    bulk_density = "./data/rawData/bulk_density.csv"
    soilComponent = "./data/rawData/soilComponent.csv"
    # 读取变量数据，拼接在一起获得输入输出变量数据
    # concatVariable(date_path[0], meto_ave_path[0], SM_15d_path[0], ST_23h_path[0], bddf.loc[0], soilComp.loc[0], savepath[0])
    # concatVariable(date_path[1], meto_ave_path[1], SM_15d_path[1], ST_23h_path[1], bddf.loc[1], soilComp.loc[1],savepath[1])
    # concatVariable(date_path[2], meto_ave_path[2], SM_15d_path[2], ST_23h_path[2], bddf.loc[2], soilComp.loc[2],savepath[2])
    # concatVariable(date_path[3], meto_ave_path[3], SM_15d_path[3], ST_23h_path[3], bddf.loc[3], soilComp.loc[3],savepath[3])
    # concatVariable(date_path[4], meto_ave_path[4], SM_15d_path[4], ST_23h_path[4], bddf.loc[4], soilComp.loc[4],savepath[4])
    # concatVariable(date_path[5], meto_ave_path[5], SM_15d_path[5], ST_23h_path[5], bddf.loc[4], soilComp.loc[4],savepath[5])
    # concatVariable(date_path[6], meto_ave_path[6], SM_15d_path[6], ST_23h_path[6], bddf.loc[4], soilComp.loc[4],savepath[6])
    # concatVariable(date_path[7], meto_ave_path[7], SM_15d_path[7], ST_23h_path[7], bddf.loc[5], soilComp.loc[5],savepath[7])
    # concatVariable(date_path[8], meto_ave_path[8], SM_15d_path[8], ST_23h_path[8], bddf.loc[5], soilComp.loc[5],savepath[8])

    # 读取6个预测点的变量数据  read_csv 不需要关闭
    dataset1 = pd.read_csv("./data/inputData1/variable_st1.csv", header=0)
    dataset2 = pd.read_csv("./data/inputData1/variable_st2.csv", header=0)
    dataset3 = pd.read_csv("./data/inputData1/variable_st3.csv", header=0)
    dataset4 = pd.read_csv("./data/inputData1/variable_st4.csv", header=0)
    dataset5_1 = pd.read_csv("./data/inputData1/variable_st51.csv", header=0)
    dataset5_2 = pd.read_csv("./data/inputData1/variable_st52.csv", header=0)
    dataset5_3 = pd.read_csv("./data/inputData1/variable_st53.csv", header=0)
    dataset6_1 = pd.read_csv("./data/inputData1/variable_st61.csv", header=0)
    dataset6_2 = pd.read_csv("./data/inputData1/variable_st62.csv", header=0)

    # 计算相关系数
    # corr1 = dataset1.corr()
    # corr2 = dataset2.corr()


    sliding_window_width = 1
    input_sequence_start = 0

    # 对每个预测点划分测试集和训练集
    # train1, test1 = split_dataset(dataset1.iloc[2:576,0:])
    # train2, test2 = split_dataset(dataset2.iloc[2:576,0:])
    # train3, test3 = split_dataset(dataset3.iloc[2:576,0:])
    # train4, test4 = split_dataset(dataset4.iloc[2:448,0:])
    # train5_1, test5_1 = split_dataset(dataset5_1.iloc[2:188,0:])
    # train5_2, test5_2 = split_dataset(dataset5_2.iloc[2:122, 0:])
    # train5_3, test5_3 = split_dataset(dataset5_3.iloc[2:25, 0:])
    # train6_1, test6_1 = split_dataset(dataset6_1.iloc[2:218,0:])
    # train6_2, test6_2 = split_dataset(dataset6_2.iloc[2:338,0:])


    # 划分X和Y 不需要滑动窗口，一行就是一条数据

    # train_L1 = np.concatenate((train1, train2,train3, train4, train5_1, train5_2, train5_3 ), axis=0)  # train6_1,train6_2
    # test_L1 = np.concatenate((test1, test2,test3,  test4, test5_1, test5_2, test5_3), axis=0)  # , test6_1,test6_2
    # train_x_L1, train_y_L1 = sliding_window(train_L1, sliding_window_width, in_start=input_sequence_start)
    # test_x_L1, test_y_L1 = sliding_window(test_L1, sliding_window_width, in_start=input_sequence_start)
    # test_x_L1 = pd.DataFrame(test_x_L1)
    # test_x_L1.to_csv('./data/test_x_L1_date.csv',encoding='utf-8_sig', index=False,header=None)
    # test_y_L1 = pd.DataFrame(test_y_L1)
    # test_y_L1.to_csv('./data/test_y_L1_date.csv',encoding='utf-8_sig', index=False,header=None)
    # train = np.concatenate((train1, train2,  train4, train5_1, train5_2, train5_3,train6_1,train6_2 ), axis=0)
    # test = np.concatenate((test1, test2,  test4, test5_1, test5_2, test5_3, test6_1,test6_2), axis=0)
    # train_x, train_y = sliding_window(train, sliding_window_width, in_start=input_sequence_start)
    # test_x, test_y = sliding_window(test, sliding_window_width, in_start=input_sequence_start)
    # test_x = pd.DataFrame(test_x)
    # test_x.to_csv('./data/test_x_date.csv',encoding='utf-8_sig', index=False,header=None)
    # test_y = pd.DataFrame(test_y)
    # test_y.to_csv('./data/test_y_date.csv',encoding='utf-8_sig', index=False,header=None)

    # ---------------------------------------------全部数据一起训练------------------------------------------
    data1 =dataset1.iloc[2:576, 1:]
    data2 = dataset2.iloc[2:576, 1:]
    data3 = dataset3.iloc[2:576, 1:]
    data4 = dataset4.iloc[2:448, 1:]
    data5_1= dataset5_1.iloc[2:188, 1:]
    data5_2= dataset5_2.iloc[2:122, 1:]
    data5_3= dataset5_3.iloc[2:25, 1:]
    data6_1= dataset6_1.iloc[2:218, 1:]
    data6_2= dataset6_2.iloc[2:338, 1:]
    np.random.seed(42)
    # 第一层不适用预测点6的数据，单独拎出来
    data_all_L1 = np.concatenate((data1, data2, data3, data4, data5_1, data5_2, data5_3), axis=0)
    shuffled_indices = np.random.permutation(len(data_all_L1))
    data_set_size = int(len(data_all_L1))
    data_indices = shuffled_indices[:data_set_size]
    data_all_L1 = data_all_L1[data_indices]
    data_all_x_L1, data_all_y_L1 = sliding_window(data_all_L1, sliding_window_width, in_start=input_sequence_start)

    data_all = np.concatenate((data1, data2, data3, data4, data5_1, data5_2, data5_3, data6_1, data6_2), axis=0)
    shuffled_indices = np.random.permutation(len(data_all))
    data_set_size = int(len(data_all))
    data_indices = shuffled_indices[:data_set_size]
    data_all = data_all[data_indices]
    data_all_x,data_all_y= sliding_window(data_all, sliding_window_width, in_start=input_sequence_start)

    train = [data_all_x, data_all_y]
    # 获取每天的数据 T为全部数据一起，F为单天数据训练
    train = dataForEachDay(train, True)

    train_L1 = [data_all_x_L1, data_all_y_L1]
    train_L1 = dataForEachDay(train_L1, True)

    # ----------------------------------------------------------------------------------------------------------
    # train = [train_x, train_y]
    # # 获取每天的数据 T为全部数据一起，F为单天数据训练,[:,1:]表示除去时间列
    # train = dataForEachDay([train_x[:,1:], train_y],True)
    #
    # train_L1 = [train_x_L1, train_y_L1]
    # train_L1 = dataForEachDay([train_x_L1[:,1:], train_y_L1], True)
    
    # 单层预测的输入数据
    # TODO 注意第一层不要使用预测点6的数据
    train_l1, scaler_trainy1 = dataForEachLayer(train_L1,level=1,day=0)
    train_l2, scaler_trainy2 = dataForEachLayer(train, level=2,day=0)
    train_l3, scaler_trainy3 = dataForEachLayer(train, 3,day=0)
    train_l4, scaler_trainy4 = dataForEachLayer(train, 4,day=0)
    train_l5, scaler_trainy5 = dataForEachLayer(train, 5,day=0)
    train_l6, scaler_trainy6 = dataForEachLayer(train, 6,day=0)
    train_l7, scaler_trainy7 = dataForEachLayer(train, 7,day=0)
    train_l8, scaler_trainy8 = dataForEachLayer(train, 8,day=0)
    train_l9, scaler_trainy9 = dataForEachLayer(train, 9,day=0)
    train_l10, scaler_trainy10 = dataForEachLayer(train, 10,day=0)




    epochs_num = 1000
    batch_size_set = 50
    verbose_set = 2  # 显示每个epoch的记录
    # lightgbm 预测
    lightgbm_model(train_l1, './save_model/lgb_model_all/lgb_level1.pkl', level=1)
    lightgbm_model(train_l2, './save_model/lgb_model_all/lgb_level2.pkl', level=2)
    lightgbm_model(train_l3, './save_model/lgb_model_all/lgb_level3.pkl', level=3)
    lightgbm_model(train_l4, './save_model/lgb_model_all/lgb_level4.pkl', level=4)
    lightgbm_model(train_l5, './save_model/lgb_model_all/lgb_level5.pkl',level=5)
    lightgbm_model(train_l6, './save_model/lgb_model_all/lgb_level6.pkl',level=6)
    lightgbm_model(train_l7, './save_model/lgb_model_all/lgb_level7.pkl',level=7)
    lightgbm_model(train_l8, './save_model/lgb_model_all/lgb_level8.pkl',level=8)
    lightgbm_model(train_l9, './save_model/lgb_model_all/lgb_level9.pkl',level=9)
    lightgbm_model(train_l10, './save_model/lgb_model_all/lgb_level10.pkl',level=10)

    # XGBoost预测
    # xgboost_model(train_l1, './save_model/xgb_model_all/xgb_level1.pkl',level=1)
    # xgboost_model(train_l2, './save_model/xgb_model_all/xgb_level2.pkl',level=2)
    # xgboost_model(train_l3, './save_model/xgb_model_all/xgb_level3.pkl',level=3)
    # xgboost_model(train_l4, './save_model/xgb_model_all/xgb_level4.pkl',level=4)
    # xgboost_model(train_l5, './save_model/xgb_model/xgb_level5.pkl',level=5)
    # xgboost_model(train_l6, './save_model/xgb_model/xgb_level6.pkl',level=6)
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
    # rf_model(train_l1, './save_model/rf_model/rf_level1.pkl', level=1)
    # rf_model(train_l2, './save_model/rf_model/rf_level2.pkl', level=2)
    # rf_model(train_l3, './save_model/rf_model/rf_level3.pkl', level=3)
    # rf_model(train_l4, './save_model/rf_model/rf_level4.pkl', level=4)
    # rf_model(train_l5, './save_model/rf_model/rf_level5.pkl', level=5)
    # rf_model(train_l6, './save_model/rf_model/rf_level6.pkl', level=6)
    # rf_model(train_l7, './save_model/rf_model/rf_level7.pkl', level=7)
    # rf_model(train_l8, './save_model/rf_model/rf_level8.pkl', level=8)
    # rf_model(train_l9, './save_model/rf_model/rf_level9.pkl', level=9)
    # rf_model(train_l10, './save_model/rf_model/rf_level10.pkl', level=10)

    print("End!!")














