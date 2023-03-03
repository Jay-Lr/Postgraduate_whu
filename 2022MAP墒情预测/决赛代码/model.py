"""
	尝试的其他模型：GRU，CNN，BILSTM等
"""
import os
import random
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from scipy.optimize import leastsq

from sklearn.utils import shuffle
from sklearn.metrics import r2_score
import sklearn.metrics as skm  # 评价指标计算
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import RepeatVector, TimeDistributed
from tensorflow.keras.models import load_model
from tensorflow import keras


from tcn import TCN, tcn_full_summary
import xgboost as xgb
import lightgbm as lgb

import pickle
import joblib  # 保存机器学习代码


def lstm_model(train, sw_width, in_start, verbose_set, epochs_num, batch_size_set):
    '''
        该函数定义 LSTM 模型
    '''
    # 使用model的callbacks函数
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath='./save_model/lstm_200_200_200_100_relu_all/LSTM.{epoch:03d}-{val_loss:.4f}-{val_accuracy:.4f}.h5',
        monitor='val_loss',  # 验证集损失函数
        verbose=1,
        save_best_only=False)  # 若设置为True，则只保存最好的模型
    # lr_reducer = keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),  # 缩放学习率的值，学习率将以lr = lr*factor的形式被减少
    #                                cooldown=0,
    #                                patience=5,
    #                                min_lr=0.5e-6)

    # callbacks = [checkpoint, lr_reducer]
    [train_x, train_y] = train
    train_x = train_x.reshape(train_x.shape[0], 1, -1)  # [n,1,84]
    train_y = train_y.reshape(train_x.shape[0], -1)  # [n,]
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

    model = Sequential()
    model.add(LSTM(200, activation='relu', return_sequences=True,  # return_sequences=True,表示是否多对多数据传递到下一个神经层网络
                   input_shape=(n_timesteps, n_features)))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(n_outputs, activation='relu'))

    model.compile(loss='mse', optimizer='adam', metrics='accuracy')
    print(model.summary())
    # 使用训练集中的10%进行验证，观察验证集的效果选择最佳模型
    history = model.fit(train_x, train_y,
                        epochs=epochs_num, batch_size=batch_size_set, verbose=verbose_set, validation_split=0.1,
                        shuffle=True,
                        callbacks=[checkpoint])
    # print('\ntrain_acc:%s' % np.mean(history.history['accuracy']), '\ntrain_loss:%s' % np.mean(history.history['loss']))

    # TODO 损失函数绘图
    # 创建一个绘图窗口
    plt.figure()
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    print(acc)
    print(val_acc)
    print(loss)
    print(val_loss)
    epochs = range(len(loss))

    plt.plot(epochs, acc, 'bo', label='Training acc')  # 'bo'为画蓝色圆点，不连线
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()  # 绘制图例，默认在右上角
    plt.savefig("Training and validation accuracy.png", dpi=300)

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig("Training and validation loss", dpi=300)
    return model

def gru_model(train, sw_width, in_start, verbose_set, epochs_num, batch_size_set):
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath='./save_model/lstm_encoder_decoder_level1/LSTM.{epoch:03d}-{val_loss:.4f}-{val_root_mean_squared_error:.4f}.h5',
        monitor='val_loss',  # 验证集损失函数
        verbose=1,
        save_best_only=False)  # 若设置为True，则只保存最好的模型
    [train_x, train_y] = train
    train_x = train_x.reshape(train_x.shape[0], sw_width, -1)  # [n,t,f]
    train_y = train_y.reshape(train_x.shape[0], -1)  # [n,t,f]
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

    inputs = keras.Input(shape=(n_timesteps, n_features))
    x = layers.GRU(100, activation='relu', return_sequences=True)(inputs)
    x = layers.Dropout(0.1)(x)  # 随机杀死神经元防止过拟合
    # 第二个GRU层
    x = layers.GRU(100, activation='relu', return_sequences=True)(x)
    x = layers.Dropout(0.1)(x)
    # 第三个GRU层
    x = layers.GRU(50, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    # 全连接层, 随机权重初始化, l2正则化
    x = layers.Dense(30, activation='relu', kernel_initializer='random_normal',
                     kernel_regularizer=keras.regularizers.l2(0.01))(x)
    # 输出层, 输入序列的10天后的股票，是时间点。保证输出层神经元个数和y_train.shape[-1]相同
    outputs = layers.Dense(n_outputs)(x)
    # 构造网络
    model = keras.Model(inputs, outputs)

    # 查看网络结构
    model.summary()
    # 构造网络
    model = keras.Model(inputs, outputs)
    # 网络编译
    model.compile(optimizer=keras.optimizers.Adam(0.01),  # adam优化器学习率0.01
                  loss=keras.losses.MeanSquaredError(),  # mse
                  metrics=[keras.metrics.RootMeanSquaredError()])  # rmse

    # 网络训练
    history = model.fit(train_x, train_y,
                            epochs=epochs_num, batch_size=batch_size_set, verbose=verbose_set, validation_split=0.2,
                            shuffle=True,
                            callbacks=[checkpoint])


    # 绘图
    # （1）查看训练信息
    history_dict = history.history  # 获取训练的数据字典
    train_loss = history_dict['loss']  # 训练集损失
    val_loss = history_dict['val_loss']  # 验证集损失
    train_rmse = history_dict['root_mean_squared_error']  # 训练集的百分比误差
    val_rmse = history_dict['val_root_mean_squared_error']  # 验证集的百分比误差

    # （2）绘制训练损失和验证损失
    plt.figure()
    plt.plot(range(epochs_num), train_loss, label='train_loss')  # 训练集损失
    plt.plot(range(epochs_num), val_loss, label='val_loss')  # 验证集损失
    plt.legend()  # 显示标签
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()

    # （3）绘制训练百分比误差和验证百分比误差
    plt.figure()
    plt.plot(range(epochs_num), train_rmse, label='train_rmse')  # 训练集损失
    plt.plot(range(epochs_num), val_rmse, label='val_rmse')  # 验证集损失
    plt.legend()  # 显示标签
    plt.xlabel('epochs')
    plt.ylabel('msle')
    plt.show()
    return


def gru_encoder_decoder(train, sw_width, in_start, verbose_set, epochs_num, batch_size_set):
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath='./save_model/lstm_encoder_decoder_level1/LSTM.{epoch:03d}-{val_loss:.4f}-{val_root_mean_squared_error:.4f}.h5',
        monitor='val_loss',  # 验证集损失函数
        verbose=1,
        save_best_only=False)  # 若设置为True，则只保存最好的模型
    [train_x, train_y] = train
    train_x = train_x.reshape(train_x.shape[0], sw_width, -1)  # [n,t,f]
    train_y = train_y.reshape(train_x.shape[0], -1,1)  # [n,t,f]
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

    inputs = keras.Input(shape=(n_timesteps, n_features))
    x = layers.GRU(200, activation='relu', return_sequences=True,dropout=0.1)(inputs)
    x = layers.GRU(100, activation='relu')(x)
    x = RepeatVector(n_outputs)(x)
    x = layers.GRU(100, activation='relu', return_sequences=True)(x)
    x = layers.GRU(50, activation='relu',return_sequences=True)(x)

    outputs = TimeDistributed(layers.Dense(1, activation='relu'))(x)
    # 构造网络
    model = keras.Model(inputs, outputs)

    # 查看网络结构
    model.summary()
    # 构造网络
    model = keras.Model(inputs, outputs)
    # 网络编译
    model.compile(optimizer=keras.optimizers.Adam(0.01),  # adam优化器学习率0.01
                  loss=keras.losses.MeanSquaredError(),  # mse
                  metrics=[keras.metrics.RootMeanSquaredError()])  # rmse

    # 网络训练
    history = model.fit(train_x, train_y,
                            epochs=epochs_num, batch_size=batch_size_set, verbose=verbose_set, validation_split=0.2,
                            shuffle=True,
                            callbacks=[checkpoint])


    # 绘图
    # （1）查看训练信息
    history_dict = history.history  # 获取训练的数据字典
    train_loss = history_dict['loss']  # 训练集损失
    val_loss = history_dict['val_loss']  # 验证集损失
    train_rmse = history_dict['root_mean_squared_error']  # 训练集的百分比误差
    val_rmse = history_dict['val_root_mean_squared_error']  # 验证集的百分比误差

    # （2）绘制训练损失和验证损失
    plt.figure()
    plt.plot(range(epochs_num), train_loss, label='train_loss')  # 训练集损失
    plt.plot(range(epochs_num), val_loss, label='val_loss')  # 验证集损失
    plt.legend()  # 显示标签
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()

    # （3）绘制训练百分比误差和验证百分比误差
    plt.figure()
    plt.plot(range(epochs_num), train_rmse, label='train_rmse')  # 训练集损失
    plt.plot(range(epochs_num), val_rmse, label='val_rmse')  # 验证集损失
    plt.legend()  # 显示标签
    plt.xlabel('epochs')
    plt.ylabel('rmse')
    plt.show()
    return

# lstm-cnn
def lstm_cnn_model(train, sw_width, in_start, verbose_set, epochs_num, batch_size_set):
    """
        lstm+cnn+Dense
    """
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath='./save_model/cnn_lstm_level1/cnn_lstm.{epoch:03d}-{val_loss:.4f}-{val_root_mean_squared_error:.4f}.h5',
        monitor='val_loss',  # 验证集损失函数
        verbose=1,
        save_best_only=False)  # 若设置为True，则只保存最好的模型
    [train_x, train_y] = train
    train_x = train_x.reshape(train_x.shape[0], sw_width, -1)  # [n,t,f]
    train_y = train_y.reshape(train_x.shape[0], -1, 1)  # [n,t,f]
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # 将每个变量视为一个通道 cnn-lstm,在特征维度卷积
    cnn =Sequential()
    cnn.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    cnn.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    cnn.add(layers.MaxPooling1D(pool_size=2))
    cnn.add(layers.Flatten())  # 一维向量

    model = Sequential()
    model.add(layers.LSTM(200, activation='relu', return_sequences=True,input_shape=(n_timesteps, n_features)))
    model.add(layers.LSTM(100, activation='relu',return_sequences=True))
    # 对每个时间步都应用相同的cnn模型
    model.add(cnn)
    model.add(layers.Dense(n_outputs,activation="relu"))

    model.compile(optimizer=keras.optimizers.Adam(0.01),  # adam优化器学习率0.01
                  loss=keras.losses.MeanSquaredError(),  # mse
                  metrics=[keras.metrics.RootMeanSquaredError()])  # rmse

    # 网络训练
    history = model.fit(train_x, train_y,
                        epochs=epochs_num, batch_size=batch_size_set, verbose=verbose_set, validation_split=0.2,
                        shuffle=True,
                        callbacks=[checkpoint])

    # （1）查看训练信息
    history_dict = history.history  # 获取训练的数据字典
    train_loss = history_dict['loss']  # 训练集损失
    val_loss = history_dict['val_loss']  # 验证集损失
    train_rmse = history_dict['root_mean_squared_error']  # 训练集的百分比误差
    val_rmse = history_dict['val_root_mean_squared_error']  # 验证集的百分比误差

    # （2）绘制训练损失和验证损失
    plt.figure()
    plt.plot(range(epochs_num), train_loss, label='train_loss')  # 训练集损失
    plt.plot(range(epochs_num), val_loss, label='val_loss')  # 验证集损失
    plt.legend()  # 显示标签
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('loss_cnn_lstm.png',dpi=300)
    # plt.show()

    # （3）绘制训练百分比误差和验证百分比误差
    plt.figure()
    plt.plot(range(epochs_num), train_rmse, label='train_rmse')  # 训练集损失
    plt.plot(range(epochs_num), val_rmse, label='val_rmse')  # 验证集损失
    plt.legend()  # 显示标签
    plt.xlabel('epochs')
    plt.ylabel('rmse')
    plt.savefig('val_cnn_lstm.png', dpi=300)
    # plt.show()
    return

# CNN 
def cnn_model(train, sw_width, in_start, verbose_set, epochs_num, batch_size_set):
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath='./save_model/cnn_level1_st/cnn_lstm.{epoch:03d}-{val_loss:.4f}-{val_root_mean_squared_error:.4f}.h5',
        monitor='val_loss',  # 验证集损失函数
        verbose=1,
        save_best_only=False)  # 若设置为True，则只保存最好的模型
    [train_x, train_y] = train
    train_x = train_x.reshape(train_x.shape[0], sw_width, -1)  # [n,t,f]
    train_y = train_y.reshape(train_x.shape[0], -1, 1)  # [n,t,f]
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # 将每个变量视为一个通道 cnn-lstm,在特征维度卷积
    cnn = Sequential()
    cnn.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu',input_shape=(n_timesteps,n_features)))
    cnn.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    cnn.add(layers.MaxPooling1D(pool_size=2))
    cnn.add(layers.Flatten())  # 一维向量

    model = Sequential()
    model.add(cnn) # 二维
    model.add(layers.Dense(100, activation="relu"))
    model.add(layers.Dense(n_outputs, activation="relu"))

    model.compile(optimizer=keras.optimizers.Adam(0.01),  # adam优化器学习率0.01
                  loss=keras.losses.MeanSquaredError(),  # mse
                  metrics=[keras.metrics.RootMeanSquaredError()])  # rmse

    # 网络训练
    history = model.fit(train_x, train_y,
                        epochs=epochs_num, batch_size=batch_size_set, verbose=verbose_set, validation_split=0.2,
                        shuffle=True,
                        callbacks=[checkpoint])

    # （1）查看训练信息
    history_dict = history.history  # 获取训练的数据字典
    train_loss = history_dict['loss']  # 训练集损失
    val_loss = history_dict['val_loss']  # 验证集损失
    train_rmse = history_dict['root_mean_squared_error']  # 训练集的百分比误差
    val_rmse = history_dict['val_root_mean_squared_error']  # 验证集的百分比误差

    # （2）绘制训练损失和验证损失
    plt.figure()
    plt.plot(range(epochs_num), train_loss, label='train_loss')  # 训练集损失
    plt.plot(range(epochs_num), val_loss, label='val_loss')  # 验证集损失
    plt.legend()  # 显示标签
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('loss_cnn_lstm.png', dpi=300)
    # plt.show()

    # （3）绘制训练百分比误差和验证百分比误差
    plt.figure()
    plt.plot(range(epochs_num), train_rmse, label='train_rmse')  # 训练集损失
    plt.plot(range(epochs_num), val_rmse, label='val_rmse')  # 验证集损失
    plt.legend()  # 显示标签
    plt.xlabel('epochs')
    plt.ylabel('rmse')
    plt.savefig('val_cnn_lstm.png', dpi=300)
    return


# 双向LSTM
def biLSTM_model(train, sw_width, in_start, verbose_set, epochs_num, batch_size_set):
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath='./save_model/biLSTM_level1/LSTM.{epoch:03d}-{val_loss:.4f}-{val_root_mean_squared_error:.4f}.h5',
        monitor='val_loss',  # 验证集损失函数
        verbose=1,
        save_best_only=False)  # 若设置为True，则只保存最好的模型
    [train_x, train_y] = train
    train_x = train_x.reshape(train_x.shape[0], sw_width, -1)  # [n,t,f]
    train_y = train_y.reshape(train_x.shape[0], -1, 1)  # [n,t,f]
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    bilstm = keras.Sequential()
    bilstm.add(layers.Bidirectional(keras.layers.LSTM(
        units=50,  # 输出维度
        input_shape=(n_timesteps, n_features),  # 输入维度
    ), merge_mode='concat'))
    bilstm.add(layers.Dense(64))
    bilstm.add(layers.LeakyReLU(alpha=0.3)) # 高级的激活函数(有自定义参数)
    bilstm.add(layers.Dense(32))
    bilstm.add(layers.LeakyReLU(alpha=0.3))
    bilstm.add(layers.Dense(15))
    bilstm.add(layers.LeakyReLU(alpha=0.3))

    bilstm.summary()
    # 定义优化器
    # nadam = keras.optimizers.Nadam(lr=1e-3)
    # 网络编译
    bilstm.compile(optimizer=keras.optimizers.Adam(0.01),  # adam优化器学习率0.01
                  loss=keras.losses.MeanSquaredError(),  # mse
                  metrics=[keras.metrics.RootMeanSquaredError()])  # rmse

    # 网络训练
    history = bilstm.fit(train_x, train_y,
                        epochs=epochs_num, batch_size=batch_size_set, verbose=verbose_set, validation_split=0.2,
                        shuffle=True,
                        callbacks=[checkpoint])

    # 绘图
    # （1）查看训练信息
    history_dict = history.history  # 获取训练的数据字典
    train_loss = history_dict['loss']  # 训练集损失
    val_loss = history_dict['val_loss']  # 验证集损失
    train_rmse = history_dict['root_mean_squared_error']  # 训练集的百分比误差
    val_rmse = history_dict['val_root_mean_squared_error']  # 验证集的百分比误差

    # （2）绘制训练损失和验证损失
    plt.figure()
    plt.plot(range(epochs_num), train_loss, label='train_loss')  # 训练集损失
    plt.plot(range(epochs_num), val_loss, label='val_loss')  # 验证集损失
    plt.legend()  # 显示标签
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()

    return

# TCN 
def tcn_model(train, sw_width, in_start, verbose_set, epochs_num, batch_size_set):
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath='./save_model/tcn_level1_st/tcn.{epoch:03d}-{val_loss:.4f}-{val_root_mean_squared_error:.4f}.h5',
        monitor='val_loss',  # 验证集损失函数
        verbose=1,
        save_best_only=False)  # 若设置为True，则只保存最好的模型
    [train_x, train_y] = train
    train_x = train_x.reshape(train_x.shape[0], sw_width, -1)  # [n,t,f]
    train_y = train_y.reshape(train_x.shape[0], -1)  # [n,t,f]
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    tcn = Sequential()
    input_layer = layers.Input(batch_shape=(batch_size_set, n_timesteps, n_features))
    tcn.add(input_layer)
    tcn.add(TCN(return_sequences=False,  # 是返回输出序列中的最后一个输出还是完整序列。如果堆叠多层则true
                ))
    tcn.add(layers.Dense(64))
    tcn.add(layers.LeakyReLU(alpha=0.3))
    tcn.add(layers.Dense(64))
    tcn.add(layers.LeakyReLU(alpha=0.3))
    tcn.add(layers.Dense(32))
    tcn.add(layers.LeakyReLU(alpha=0.3))
    tcn.add(layers.Dense(15))
    tcn.add(layers.LeakyReLU(alpha=0.3))
    tcn.compile('adam', loss='mse', metrics=['accuracy'])
    tcn.summary()

    tcn.compile(optimizer=keras.optimizers.Adam(0.01),  # adam优化器学习率0.01
                   loss=keras.losses.MeanSquaredError(),  # mse
                   metrics=[keras.metrics.RootMeanSquaredError()])  # rmse

    # 网络训练
    history = tcn.fit(train_x, train_y,
                         epochs=epochs_num, batch_size=batch_size_set, verbose=verbose_set, validation_split=0.2,
                         shuffle=True,
                         callbacks=[checkpoint])

    plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.savefig('Training and validation loss.png', dpi=300)
    plt.legend()
    return

# encoder-decoder lstm
def lstm_initial_state(train, sw_width, in_start, verbose_set, epochs_num, batch_size_set):
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath='./save_model/lstm_initial_state_level1_st/LSTM.{epoch:03d}-{val_loss:.4f}-{val_root_mean_squared_error:.4f}.h5',
        monitor='val_loss',  # 验证集损失函数
        verbose=1,
        save_best_only=False)  # 若设置为True，则只保存最好的模型
    [train_x, train_y] = train
    train_x = train_x.reshape(train_x.shape[0], sw_width, -1)  # [n,t,f]
    train_y = train_y.reshape(train_x.shape[0], -1, 1)  # [n,t,f]
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    encoder_inputs = layers.Input(shape=(sw_width, n_features))
    encoder = Sequential()
    encoder.add(layers.LSTM(128, activation='relu', return_sequences=True))
    encoder.add(layers.LSTM(128,activation="relu",return_state=True))
    encoder_outputs,state_h,state_c = encoder(encoder_inputs)
    encoder_states = [state_h,state_c]
    # define training decoder  训练时用的是真值
    decoder_inputs = layers.Input(shape=(None,n_outputs))
    decoder = Sequential()
    decoder.add(layers.LSTM(128,return_sequences=True,return_state=True))
    decoder.add(layers.LSTM(128, return_sequences=True, return_state=True))
    decdoer_outputs,_,_=decoder(decoder_inputs,initial_state=encoder_states)
    decoder_dense = layers.Dense(n_outputs,activation="relu")
    decoder_outputs = decoder_dense(decdoer_outputs)

    model = keras.Model([encoder_inputs,decoder_inputs],decoder_outputs)
    # define inference encoder
    encoder_model = keras.Model(encoder_inputs, encoder_states)
    # define inference decoder
    decoder_state_input_h = layers.Input(shape=(128,))
    decoder_state_input_c = layers.Input(shape=(128,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = keras.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    model.compile(optimizer=keras.optimizers.Adam(0.01), loss='mse', metrics=[keras.metrics.RootMeanSquaredError()])
    print(model.summary())

    history = model.fit(train_x, train_y,
                        epochs=epochs_num, batch_size=batch_size_set, verbose=verbose_set, validation_split=0.2,
                        shuffle=True,
                        callbacks=[checkpoint])

    plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.savefig('Training and validation loss.png', dpi=300)
    plt.legend()
    return model,encoder_model,decoder_model


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


# lightgbm
def lightgbm_model(train, path, level):
    # 训练集
    [train_x, train_y] = train
    train_y = np.ravel(train_y)
    # lightgbm模型参数设置
    cv_params = {
        # 'n_estimators':np.arange(1000,2000,100),
        # 'max_depth': [5, 7, 9,12,15,19],
        # 'num_leaves': np.arange(10, 30, 2),
    }
    """
        首先确定较大的学习率，0.1
        调整 n_estimators，max_depth，num_leaves，colsample_bytree，subsample，bagging_freq
        确定L1L2正则reg_alpha和reg_lambda
        降低调整学习率 
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
            'n_estimators': 3000,  # 分类器数量 5000
            'learning_rate': 0.1,  # 学习速率  [0.01, 0.015, 0.025, 0.05, 0.1]
            'max_depth': 9,  # 树的最大深度 [3, 5, 6, 7, 9, 12, 15, 17, 25]
            'num_leaves': 22,  # 一般设置为(0, 2^max_depth - 1]的一个数值
            'colsample_bytree': 0.6,  # 建树的特征选择比例 [0.6, 0.7, 0.8, 0.9, 1]
            'subsample': 1,  # 建树的样本采样比例 [0.6, 0.7, 0.8, 0.9, 1]
            'bagging_freq': 2,  # k 意味着每 k 次迭代执行bagging  [2, 4, 5, 6, 8]
            'reg_alpha': 0,
            'reg_lambda': 0,
        }
    elif (level == 2):
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'mse',
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 7,
            'num_leaves': 12,
            'colsample_bytree': 0.9,
            'subsample': 0.9,
            'bagging_freq': 2,
            'reg_alpha': 0,
            'reg_lambda': 36,
        }
    elif (level == 3):
        params = {
            'boosting_type': 'gbdt',  # 使用的树的模型：梯度提升决策树（GBDT）
            'objective': 'regression',  # 目标函数 使用L2正则项的回归模型（默认值)
            'metric': 'mse',  # 评估函数
            'n_estimators': 200,  # 分类器数量
            'learning_rate': 0.04,  # 学习速率  [0.01, 0.015, 0.025, 0.05, 0.1]
            'max_depth': 7,  # 树的最大深度 [3, 5, 6, 7, 9, 12, 15, 17, 25]
            'num_leaves': 10,  # 一般设置为(0, 2^max_depth - 1]的一个数值
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
            'n_estimators': 335,  # 分类器数量
            'learning_rate': 0.03,  # 学习速率  [0.01, 0.015, 0.025, 0.05, 0.1]
            'max_depth': 3,  # 树的最大深度 [3, 5, 6, 7, 9, 12, 15, 17, 25]
            'num_leaves': 5,  # 一般设置为(0, 2^max_depth - 1]的一个数值
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
            'n_estimators': 1000,  # 分类器数量
            'learning_rate': 0.08,  # 学习速率  [0.01, 0.015, 0.025, 0.05, 0.1]
            'max_depth': 12,  # 树的最大深度 [3, 5, 6, 7, 9, 12, 15, 17, 25]
            'num_leaves': 28,  # 一般设置为(0, 2^max_depth - 1]的一个数值
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
    joblib.dump(optimized_GBM, path)
    # df1 = pd.DataFrame()
    # df1['feature name'] = ['SM_t-2','SM_t-1','bulk density', 'sand', 'silt', 'clay', '相对湿度(%)', '大气压力(hPa)',
    #                        '风速(m/s)', '风向(°)', '雨量(mm)', '当前太阳辐射强度(W/m²)', '累计太阳辐射量(MJ/m²)',
    #                        '空气温度(℃)', 'day']
    # df1['importance'] = model.feature_importances_
    # df1.sort_values(by='importance', inplace=True, ascending=False)
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
    # joblib.dump(optimized_GBM, path)
    return