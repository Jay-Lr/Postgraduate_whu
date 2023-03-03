"""
    lstm,lgb,xgb,rf->预测待提交的6个预测点的数据
    LSTM数据处理：
        sliding_window()
    LGB,XGB等数据处理：
        dataForEachDay(),dataForEachLayer()
    模型预测和评价：
        forecastSubmitData(),evaluate_forecasts(),evaluate_forecasts1()
    
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as skm
from scipy.optimize import leastsq
import matplotlib
from sklearn.preprocessing import MinMaxScaler
import scipy.interpolate as spi
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import RepeatVector, TimeDistributed
import math
from tensorflow.keras.models import load_model
from tensorflow import keras
import joblib
import os



# ---------------------------------------------------提交数据预测--------------------------------------------------------------

def forecastSubmitData(model,modelName,submit_x1,level,point):
    x= submit_x1
    if(modelName=='lstm'): # LSTM多天单层
        X = sliding_window(x,15,15,0,level=level)
        min_max = pd.read_csv('./lstmData/min_max_level'+str(level)+'_all.csv', header=None)
        x_max = np.array(min_max.iloc[:,0])
        x_min = np.array(min_max.iloc[:, 1])
        y_max = np.array(min_max.iloc[:,2])
        y_min = np.array(min_max.iloc[:,3])
        X = X.reshape(X.shape[0], -1) # 便于归一化
        X = (X - x_min) / (x_max - x_min)
        X = X.reshape(X.shape[0],15, -1)
        yhat = model.predict(X, verbose=0)
        yhat = yhat.reshape(yhat.shape[0],15)
        predicts = yhat * (y_max[0:15] - y_min[0:15]) + y_min[0:15]
        predicts = predicts.reshape(15,1)
        if (level > 1):
            data = pd.read_csv('./lstmData/submitData/predictions_' + modelName +str(point)+ '.csv', header=None)
            data['new'] = predicts
            pd.DataFrame(data).to_csv('./lstmData/submitData/predictions_' + modelName +str(point)+ '.csv', encoding='utf-8_sig',index=False, header=None)
        else:
            pd.DataFrame(predicts).to_csv('./lstmData/submitData/predictions_' + modelName + str(point)+ '.csv', encoding='utf-8_sig',index=False, header=None)


    else: # 其他ML模型 多天单层
        X = dataForEachDay(x, True)
        X, scaler_y = dataForEachLayer(X, level, day=0)
        yhat = model.predict(X)
        yhat = yhat.reshape(yhat.shape[0], -1)
        predicts = yhat * (scaler_y[0] - scaler_y[1]) + scaler_y[1]
        # 每天的数据 是直接append的,转置时，按列转置
        predictss = predicts.reshape(-1, 15, order='f')  # (n,15) 每天对应的后面15天的数据 【15，15】
        pd.DataFrame(predictss).to_csv('./data/submitData/predictions_' + modelName +str(point)+"_"+str(level)+ '.csv', encoding='utf-8_sig',index=False, header=None)


    return

def sliding_window(data,sw_width, n_out, in_start,level):
    data = np.array(data) # [15,f] => [1,15,f]
    X = []
    # 取出level层的土壤数据
    if (level != 0):
        data = np.c_[data[:, 0:19], data[:, 19 + level - 1], data[:, 29 + level - 1], data[:, 39 + level - 1], data[:,49 + level - 1], data[:,59 + level - 1], data[:,69 + level - 1]]
    for _ in range(len(data)):
        in_end = in_start + sw_width
        out_end = in_end + n_out

        # 保证截取样本完整，最大元素索引不超过原序列索引，则截取数据；否则丢弃该样本
        if in_end <= len(data):
            # 训练数据以滑动步长1截取
            data_seq = data[in_start:in_end, :]  # (15,f)
            X.append(data_seq)

        in_start += 1  # 滑动步长增加

    # data = data.reshape((1,15, data.shape[1]))  # 二维 n,f
    return np.array(X)
# ------------------------------------------------------------------------------------------------------------------------


# 评价真值和预测值
def evaluate_forecasts1(actual, predicted,pre_length):
    '''
        scores 每15天滚动预测 每个深度的的均方差
        R2 每15天滚动预测 每个深度的R2
        score 每天预测误差 的均方之和/样本/15天
    '''
    scores = list()
    R2 = []

    # [n,10] 分别计算每个深度的误差
    # 每15天样本的平均误差
    for m in range(int(actual.shape[0]/pre_length)):
        start = m * pre_length
        end = start + pre_length
        r2 = []
        rmse = []
        for i in range(actual.shape[1]):
            mse = skm.mean_squared_error(actual[start:end, i], predicted[start:end, i])
            rmse.append(math.sqrt(mse))
            r2.append(skm.r2_score(actual[start:end, i], predicted[start:end, i]))
        scores.append(np.array(rmse))
        R2.append(np.array(r2))

    s = 0
    score = [] # 每个样本10个深度总误差/样本数/深度，平均每个样本每个深度的误差
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
        if((row+1)/15==0):
            score.append(math.sqrt(s / (15 * actual.shape[1])))

    scores = np.array(scores)
    print('\n每15天滚动预测 每个深度的R2:',R2)
    print('\n每15天滚动预测 每个深度的RMSE:', scores)
    print('\n每天滚动预测 所有深度的RMSE:', score)
    np.savetxt('./evaluation/scores.csv',scores , delimiter=',')
    return score, scores

def evaluate_forecasts(actual, predicted):
    '''
        scores 每天 所有样本 的均方差
        score 每个样本 每天预测误差 的均方之和/样本，平均每个样本每个步长的误差
    '''
    scores = list()
    R2=[]
    # 所有样本，每层土壤湿度的预测rmse
    for i in range(actual.shape[1]):
        mse = skm.mean_squared_error(actual[:, i], predicted[:, i])
        rmse = math.sqrt(mse)
        scores.append(rmse)
        # TODO　计算R2 多输出
        R2.append(skm.r2_score(actual[:,i], predicted[:,i]))


    s = 0  # 计算总的 RMSE
    # 平均每个样本，每层土壤湿度的预测误差
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = math.sqrt(s / (actual.shape[0] * actual.shape[1]))
    print('actual.shape[0]:{}, actual.shape[1]:{}'.format(actual.shape[0], actual.shape[1]))

    print('\n 每个深度的R2:', R2)
    print('\n每个深度的RMSE:', scores)
    print('\n所有深度的RMSE:', score)
    scores = np.array(scores)
    # np.savetxt('./evaluation/scores.csv', scores, delimiter=',')



    fig = plt.figure( figsize=(8, 10))  # figsize=(10, 6), dpi=600
    for i in range(0, actual.shape[1],2): # actual.shape[1],
        plt.subplot(actual.shape[1], 1, i+1)
        feature_name = '第'+str(i+1)+'层'+'土壤水分含量'
        plt.plot(actual[:,i], c='orange', linewidth=0.1,label='真值')
        plt.plot(predicted[:,i], c='g',linewidth=0.1, label='预测值')
        plt.rcParams.update({'font.size': 4})
        plt.title(feature_name, size=5)  # y=0,
        plt.grid(linestyle='--', alpha=0.5)
        plt.xticks(fontsize=4)
        plt.yticks(fontsize=4)
        if(i==0):
            # plt.ylabel('水分含量(%)',fontsize=5)
            plt.legend()
        # fig.tight_layout()
        plt.subplots_adjust(left=0.1, bottom=0.1, right=None, top=None,wspace=None, hspace=0.1)

     # 调整整体空白
    fig.savefig('Predict and truth2.png',dpi=600)
    return score, scores


# 划分每层数据
def dataForEachLayer(data,level,day):
    """
    :param x: 数组x
    :param y: 数组y
    :param level: 土壤层数
    :return:
    """
    x = data
    # day=0 全部天一起训练，预测15天
    if (day != 0):
        # 每天单独训练-只预测1天
        x = x[:, :, day - 1]
    x_date = x[:, 0:2]
    x_bd = x[:, 2 + level - 1]
    x_sand = x[:, 12 + level - 1]
    x_silt = x[:, 22 + level - 1]
    x_clay = x[:, 32 + level - 1]
    x_st = np.c_[x[:, 42], x[:, 43 + level - 1]]  # 地表温度和level层温度
    x_t_2 = x[:, 53 + level - 1]
    x_t_1 = x[:, 63 + level - 1]
    x_meteo = x[:, 73:]
    X = np.c_[x_date, x_bd, x_sand, x_silt, x_clay, x_st, x_t_2, x_t_1, x_meteo]  # 全部天一起训练时，多出了一个 天数 变量

    # 先划分再归一化,输入变量为15个，输出1个
    min_max = pd.read_csv('./data/submitData/min_max_st_all.csv', header=None)
    x_max = np.array(min_max.iloc[19 * (level - 1):level * 19, 0])
    x_min = np.array(min_max.iloc[19 * (level - 1):level * 19, 1])
    y_max = np.array(min_max.iloc[19 * (level - 1):19 * (level - 1) + 1, 2])
    y_min = np.array(min_max.iloc[19 * (level - 1):19 * (level - 1) + 1, 3])
    # 数组广播 完成 加减乘除
    X = (X - x_min) / (x_max - x_min)

    # data, scaler_y = scalerVariable(X, Y, 0, 0)
    return X, [y_max, y_min]

# 将原始数据拆分为15天同时预测的数据集
def dataForEachDay(data,flag):
    """
    :param data:
    :param flag: 为T,15天的数据按行拼接在一起；为F,提取出每天的数据，得到X，Y为三维数组，[samples,features,days]
    :return:
    """
    x = np.array(data)[:,0:-150]
    x1 = np.concatenate([x[:, 0:53], x[:, 173:193]], axis=1)  # 土壤组成数据,温度数据,历史墒情
    X = []
    Y = []
    for i in range(15):
        x2 = x[:, (53 + i * 8):(53 + (i + 1) * 8)]
        # yy = y[:, i * 10:(i + 1) * 10]
        xx = np.concatenate([x1, x2], axis=1)
        if (flag):
            # 为每天的数据增加一个天数变量
            xx = pd.DataFrame(xx)
            xx['day'] = i + 1
            xx = np.array(xx)
            #  15天的要一起计算，则把15天的数据拼在一起
            if (i == 0):
                X = xx
                # Y = yy
            else:
                X = np.append(X, xx, axis=0)  # 沿行拼接在一起
                # Y = np.append(Y, yy, axis=0)
        else:
            # 15天，每天为一条数据
            if (i == 0):
                X = xx
                # Y = yy
            elif (i == 1):
                X = np.stack([X, xx], axis=-1)
                # Y = np.stack([Y, yy], axis=-1)
            else:
                X = np.append(X, xx.reshape(xx.shape[0], -1, 1), axis=2)
                # Y = np.append(Y, yy.reshape(yy.shape[0], -1, 1), axis=2)
    return X


sliding_window_width = 1
input_sequence_start = 0

# 六个预测点数据
submit_x1= pd.read_csv("./lstmData/submitData/variable1.csv", header=0)
submit_x2= pd.read_csv("./lstmData/submitData/variable2.csv", header=0)
submit_x3= pd.read_csv("./lstmData/submitData/variable3.csv", header=0)
submit_x4= pd.read_csv("./lstmData/submitData/variable4.csv", header=0)
submit_x5= pd.read_csv("./lstmData/submitData/variable5.csv", header=0)
submit_x6= pd.read_csv("./lstmData/submitData/variable6.csv", header=0)
# submit_x1= pd.read_csv("./data/submitData/variable1.csv", header=0)
# submit_x2= pd.read_csv("./data/submitData/variable2.csv", header=0)
# submit_x3= pd.read_csv("./data/submitData/variable3.csv", header=0)
# submit_x4= pd.read_csv("./data/submitData/variable4.csv", header=0)
# submit_x5= pd.read_csv("./data/submitData/variable5.csv", header=0)
# submit_x6= pd.read_csv("./data/submitData/variable6.csv", header=0)

modelName='lstm'

# lstm预测
model1 = load_model('./save_model/lstm_encoder_decoder_level1/LSTM.764-0.0005-0.0226.h5')
model2 = load_model('./save_model/lstm_encoder_decoder_level2/LSTM.979-0.0002-0.0145.h5')
model3 = load_model('./save_model/lstm_encoder_decoder_level3/LSTM.915-0.0002-0.0123.h5')
model4 = load_model('./save_model/lstm_encoder_decoder_level4/LSTM.837-0.0003-0.0161.h5')
model5 = load_model('./save_model/lstm_encoder_decoder_level5/LSTM.922-0.0001-0.0118.h5')
model6 = load_model('./save_model/lstm_encoder_decoder_level6/LSTM.985-0.0001-0.0106.h5')
model7 = load_model('./save_model/lstm_encoder_decoder_level7/LSTM.954-0.0001-0.0098.h5')
model8 = load_model('./save_model/lstm_encoder_decoder_level8/LSTM.963-0.0000-0.0053.h5')
model9 = load_model('./save_model/lstm_encoder_decoder_level9/LSTM.995-0.0000-0.0050.h5')
model10 = load_model('./save_model/lstm_encoder_decoder_level10/LSTM.836-0.0000-0.0046.h5')

# rf预测
# model1 = joblib.load('./save_model/rf_model_all/rf_level1.pkl')
# model2 = joblib.load('./save_model/rf_model_all/rf_level2.pkl')
# model3 = joblib.load('./save_model/rf_model_all/rf_level3.pkl')
# model4 = joblib.load('./save_model/rf_model_all/rf_level4.pkl')
# model5 = joblib.load('./save_model/rf_model_all/rf_level5.pkl')
# model6 = joblib.load('./save_model/rf_model_all/rf_level6.pkl')
# model7 = joblib.load('./save_model/rf_model_all/rf_level7.pkl')
# model8 = joblib.load('./save_model/rf_model_all/rf_level8.pkl')
# model9 = joblib.load('./save_model/rf_model_all/rf_level9.pkl')
# model10 = joblib.load('./save_model/rf_model_all/rf_level10.pkl')

# lightgbm预测
# model1 = joblib.load('./save_model/lgb_model_all/lgb_level1.pkl')
# model2 = joblib.load('./save_model/lgb_model_all/lgb_level2.pkl')
# model3 = joblib.load('./save_model/lgb_model_all/lgb_level3.pkl')
# model4 = joblib.load('./save_model/lgb_model_all/lgb_level4.pkl')
# model5 = joblib.load('./save_model/lgb_model_all/lgb_level5.pkl')
# model6 = joblib.load('./save_model/lgb_model_all/lgb_level6.pkl')
# model7 = joblib.load('./save_model/lgb_model_all/lgb_level7.pkl')
# model8 = joblib.load('./save_model/lgb_model_all/lgb_level8.pkl')
# model9 = joblib.load('./save_model/lgb_model_all/lgb_level9.pkl')
# model10 = joblib.load('./save_model/lgb_model_all/lgb_level10.pkl')

# xgboost 预测
# model1 = joblib.load('./save_model/xgb_model_all/xgb_level1.pkl')
# model2 = joblib.load('./save_model/xgb_model_all/xgb_level2.pkl')
# model3 = joblib.load('./save_model/xgb_model_all/xgb_level3.pkl')
# model4 = joblib.load('./save_model/xgb_model_all/xgb_level4.pkl')
# model5 = joblib.load('./save_model/xgb_model_all/xgb_level5.pkl')
# model6 = joblib.load('./save_model/xgb_model_all/xgb_level6.pkl')
# model7 = joblib.load('./save_model/xgb_model_all/xgb_level7.pkl')
# model8 = joblib.load('./save_model/xgb_model_all/xgb_level8.pkl')
# model9 = joblib.load('./save_model/xgb_model_all/xgb_level9.pkl')
# model10 = joblib.load('./save_model/xgb_model_all/xgb_level10.pkl')



# 预测15天
forecastSubmitData(model1,modelName,submit_x1.iloc[:,1:],level=1,point=1) # 去掉时间列
forecastSubmitData(model2,modelName,submit_x6.iloc[:,1:],level=2,point=6)
forecastSubmitData(model3,modelName,submit_x6.iloc[:,1:],level=3,point=6)
forecastSubmitData(model4,modelName,submit_x6.iloc[:,1:],level=4,point=6)
forecastSubmitData(model5,modelName,submit_x6.iloc[:,1:],level=5,point=6)
forecastSubmitData(model6,modelName,submit_x6.iloc[:,1:],level=6,point=6)
forecastSubmitData(model7,modelName,submit_x6.iloc[:,1:],level=7,point=6)
forecastSubmitData(model8,modelName,submit_x6.iloc[:,1:],level=8,point=6)
forecastSubmitData(model9,modelName,submit_x6.iloc[:,1:],level=9,point=6)
forecastSubmitData(model10,modelName,submit_x6.iloc[:,1:],level=10,point=6)

print('end!')






















