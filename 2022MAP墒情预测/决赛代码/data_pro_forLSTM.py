"""
    lstm等模型 多天多层预测，滑动窗口 获取数据集
    输入：土壤组成，气象，t+15气象，墒情
    输出：t+15墒情
    
    interpolate:缺失变量插值
    readrawData:处理原始数据（除去气象数据缺失的预测点3，删去风速和风向缺失的预测点4）
    concatVariable:变量拼接
    sliding_window: 滑动数据集
    split_dataset:划分训练，测试集
    
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
from tensorflow.keras import layers
from tensorflow.keras.layers import RepeatVector, TimeDistributed
import math
from sklearn.metrics import r2_score
from tensorflow.keras.models import load_model
from tensorflow import keras
from sklearn.utils import shuffle
from tcn import TCN, tcn_full_summary
# 绘制深度学习结构图
from tensorflow.keras.utils import plot_model

# 中文字体显示
plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

def lstm_model(train, sw_width, in_start, verbose_set, epochs_num, batch_size_set):
    '''
        该函数定义 LSTM 模型
    '''
    # 使用model的callbacks函数
    checkpoint = keras.callbacks.ModelCheckpoint(filepath='./save_model/lstm_200_200_200_100_relu_st_level1/LSTM.{epoch:03d}-{val_loss:.4f}-{val_accuracy:.4f}.h5',
                                                 monitor='val_loss',  # 验证集损失函数
                                                 verbose=1,
                                                 save_best_only=False)  # 若设置为True，则只保存最好的模型
    # lr_reducer = keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),  # 缩放学习率的值，学习率将以lr = lr*factor的形式被减少
    #                                cooldown=0,
    #                                patience=5,
    #                                min_lr=0.5e-6)

    # callbacks = [checkpoint, lr_reducer]
    [train_x,train_y] = train # X-> [N,15,F], y-> [N,15,10]
    train_x = train_x.reshape(train_x.shape[0],sw_width,-1)  # [n,1,84]
    train_y = train_y.reshape(train_x.shape[0],-1)  # [n,]
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

    model = Sequential()
    model.add(LSTM(200, activation='relu', return_sequences=True,  # return_sequences=True,表示是否多对多数据传递到下一个神经层网络，如果只关注最后一个时间步的输出，选择return_sequences=false
                   input_shape=(n_timesteps, n_features)))
    model.add(LSTM(200, activation='relu',return_sequences=True))
    model.add(LSTM(200, activation='relu',return_sequences=True))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(n_outputs,activation='relu'))

    model.compile(loss='mse', optimizer='adam',metrics='accuracy')
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
    plt.savefig("Training and validation accuracy.png",dpi=300)

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig("Training and validation loss",dpi=300)
    return model

# 编码-解码LSTM
def encoder_decoder_lstm(train, sw_width, in_start, verbose_set, epochs_num, batch_size_set):
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath='./save_model/lstm_encoder_decoder_level1/LSTM.{epoch:03d}-{val_loss:.4f}-{val_root_mean_squared_error:.4f}.h5',
        monitor='val_loss',  # 验证集损失函数
        verbose=1,
        save_best_only=False)  # 若设置为True，则只保存最好的模型
    [train_x, train_y] = train
    train_x = train_x.reshape(train_x.shape[0], sw_width, -1)  # [n,t,f]
    train_y = train_y.reshape(train_x.shape[0], -1,1)  #[n,t,f]
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    model = Sequential()
    # 编码器
    model.add(LSTM(200, activation='relu',input_shape=(sw_width, n_features),return_sequences=True)) # 默认，return_sequences =False,即只输出了一个时间步
    model.add(LSTM(128, activation='relu',input_shape=(sw_width, n_features)))  # 默认，return_sequences =False,即只输出了一个时间步
    # RepeatVector，将输出转为相同时间步的输出序列
    model.add(RepeatVector(n_outputs))
    # 解码器
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    # 全连接层 TimeDistributed的特定层，它将相同的全连接层应用于每个时间步，由于每个时间步只有一个输出，因此一个时间步的dense应该为1
    model.add(TimeDistributed(Dense(1)))  # 输出维度 [n,t]
    model.compile(optimizer='adam', loss='mse', metrics=[keras.metrics.RootMeanSquaredError()])
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
    plt.savefig('Training and validation loss.png',dpi=300)
    plt.legend()
    return


def gru_model(train, sw_width, in_start, verbose_set, epochs_num, batch_size_set):
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath='./save_model/gru_level1_st/gru.{epoch:03d}-{val_loss:.4f}-{val_root_mean_squared_error:.4f}.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=False)  # 若设置为True，则只保存最好的模型
    [train_x, train_y] = train
    train_x = train_x.reshape(train_x.shape[0], sw_width, -1)  # [n,t,f]
    train_y = train_y.reshape(train_x.shape[0], -1)  # [n,t,f]
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

    inputs = keras.Input(shape=(n_timesteps, n_features))
    x = layers.GRU(200, activation='relu', return_sequences=True)(inputs)
    x = layers.Dropout(0.1)(x)  # 随机杀死神经元防止过拟合
    # 第二个GRU层
    x = layers.GRU(200, activation='relu', return_sequences=True)(x)
    x = layers.Dropout(0.1)(x)
    # 第三个GRU层
    x = layers.GRU(200, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    # 全连接层, 随机权重初始化, l2正则化
    x = layers.Dense(100, activation='relu', kernel_initializer='random_normal',
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
    plt.ylabel('rmse')
    plt.show()
    return

def gru_encoder_decoder(train, sw_width, in_start, verbose_set, epochs_num, batch_size_set):
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath='./save_model/gru_encoder_decoder_level1/gru.{epoch:03d}-{val_loss:.4f}-{val_root_mean_squared_error:.4f}.h5',
        monitor='val_loss',  # 验证集损失函数
        verbose=1,
        save_best_only=False)  # 若设置为True，则只保存最好的模型
    [train_x, train_y] = train
    train_x = train_x.reshape(train_x.shape[0], sw_width, -1)  # [n,t,f]
    train_y = train_y.reshape(train_x.shape[0], -1,1)  # [n,t,f]
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

    inputs = keras.Input(shape=(n_timesteps, n_features))
    x = layers.GRU(200, activation='relu', return_sequences=True,dropout=0.1)(inputs)
    x = layers.GRU(200, activation='relu')(x)
    x = RepeatVector(n_outputs)(x)
    # 第二个GRU层
    x = layers.GRU(200, activation='relu', return_sequences=True)(x)
    x = layers.GRU(100, activation='relu',return_sequences=True)(x)

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
    plt.ylabel('msle')
    plt.show()
    return


def biLSTM_model(train, sw_width, in_start, verbose_set, epochs_num, batch_size_set):
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath='./save_model/biLSTM4_level1/biLSTM.{epoch:03d}-{val_loss:.4f}-{val_root_mean_squared_error:.4f}.h5',
        monitor='val_loss',  # 验证集损失函数
        verbose=1,
        save_best_only=False)  # 若设置为True，则只保存最好的模型
    [train_x, train_y] = train
    train_x = train_x.reshape(train_x.shape[0], sw_width, -1)  # [n,t,f]
    train_y = train_y.reshape(train_x.shape[0],-1,1)  # [n,t,f]
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    bilstm = Sequential()
    bilstm.add(layers.Bidirectional(keras.layers.LSTM(units=200, input_shape=(n_timesteps, n_features),return_sequences=True), merge_mode='concat'))
    bilstm.add(layers.Bidirectional(keras.layers.LSTM(units=200,return_sequences=True), merge_mode='concat'))
    bilstm.add(layers.Bidirectional(keras.layers.LSTM(units=128,return_sequences=True), merge_mode='concat'))
    bilstm.add(layers.Bidirectional(keras.layers.LSTM(units=128, return_sequences=True),merge_mode='concat'))
    # bilstm.add(layers.LSTM(128, activation='relu', return_sequences=True))
    bilstm.add(TimeDistributed(Dense(1)))
    # bilstm.add(layers.Dense(15))
    # bilstm.add(layers.LeakyReLU(alpha=0.3))
    # 需要先Build再summary
    bilstm.build(input_shape=(batch_size_set,n_timesteps, n_features))
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
    history_dict = history.history  # 获取训练的数据字典
    train_loss = history_dict['loss']  # 训练集损失
    val_loss = history_dict['val_loss']  # 验证集损失
    train_rmse = history_dict['root_mean_squared_error']  # 训练集的百分比误差
    val_rmse = history_dict['val_root_mean_squared_error']  # 验证集的百分比误差

    plt.figure()
    plt.plot(range(epochs_num), train_loss, label='train_loss')  # 训练集损失
    plt.plot(range(epochs_num), val_loss, label='val_loss')  # 验证集损失
    plt.legend()  # 显示标签
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('Training and validation loss.png', dpi=300)

    plt.figure()
    plt.plot(range(epochs_num), train_rmse, label='train_rmse')  # 训练集损失
    plt.plot(range(epochs_num), val_rmse, label='val_rmse')  # 验证集损失
    plt.legend()  # 显示标签
    plt.xlabel('epochs')
    plt.ylabel('rmse')
    plt.savefig('Training and validation rmse.png', dpi=300)

    return


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


def tcn_model(train, sw_width, in_start, verbose_set, epochs_num, batch_size_set):
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath='./save_model/tcn_model_level1_st/tcn.{epoch:03d}-{val_loss:.4f}-{val_root_mean_squared_error:.4f}.h5',
        monitor='val_loss',  # 验证集损失函数
        verbose=1,
        save_best_only=False)  # 若设置为True，则只保存最好的模型
    [train_x, train_y] = train
    train_x = train_x.reshape(train_x.shape[0], sw_width, -1)  # [n,t,f]
    train_y = train_y.reshape(train_x.shape[0], -1)  # [n,t,f]
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    tcn = Sequential()
    input_layer = layers.Input(input_shape=( n_timesteps, n_features))
    tcn.add(input_layer)
    tcn.add(TCN(
                return_sequences=True,  # 是返回输出序列中的最后一个输出还是完整序列。 如果堆叠多层则true
                activation='relu',  # 残差块中使用的激活函数 o = Activation(x + F(x)).
                name='tcn1'  # 使用多个TCN时，要使用唯一的名称
                ))
    tcn.add(TCN(return_sequences=False,  # 是返回输出序列中的最后一个输出还是完整序列。 如果堆叠多层则true
                name='tcn2'  # 使用多个TCN时，要使用唯一的名称
                ))
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


# 最大最小归一化
def scalerVariable(x,y,m,n):
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    X = scaler_x.fit_transform(x.reshape(x.shape[0],-1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    Y = scaler_y.fit_transform(y.reshape(y.shape[0],-1))
    X1 = X.reshape(x.shape[0],x.shape[1],-1)
    Y1 = Y.reshape(y.shape[0], y.shape[1], -1)
    # 注意为常量的bulk density和土壤组成 不进行缩放  24：63列
    # X[:, m:n] = x[:, m:n]
    # 保存所有数各个列的最大最小值，用于滚动预测
    # df = pd.concat([pd.DataFrame(x.reshape(x.shape[0],-1)).max(),pd.DataFrame(x.reshape(x.shape[0],-1)).min(),pd.DataFrame(y.reshape(y.shape[0],-1)).max(),pd.DataFrame(y.reshape(y.shape[0],-1)).min()],axis=1)
    # df.to_csv('./lstmData/min_max_level2_st.csv',encoding='utf-8_sig', index=False,header=None)
    return [X1,Y1],scaler_y

# 滑动窗口
def sliding_window(data, sw_width, n_out, in_start,a,b,level):
    '''
        该函数实现窗口宽度为45(输入30，输出15)、滑动步长为1天的滑动窗口截取序列数据
        a,b是 取的读取数据中可以进行窗口滑动的范围
        level 是如果单层训练的话，获取单层数据
    '''
    # 增加月和年中天数据

    data = np.array(data.iloc[a:b,1:])
    data = data.reshape((data.shape[0], data.shape[1]))  #  二维 n,f
    X, y = [], []
    # t取出level层的土壤数据
    if(level!=0):
        data = np.c_[data[:,0:19],data[:,19+level-1],data[:,29+level-1],data[:,39+level-1],data[:,49+level-1],data[:,59+level-1],data[:,69+level-1]]
        # data = np.c_[data[:, 0:16], data[:, 16 + level - 1], data[:, 26 + level - 1], data[:, 36 + level - 1], data[:,46 + level - 1], data[:,56 + level - 1]]
    for _ in range(len(data)):
        in_end = in_start + sw_width
        out_end = in_end + n_out

        # 保证截取样本完整，最大元素索引不超过原序列索引，则截取数据；否则丢弃该样本
        if out_end < len(data):
            # 训练数据以滑动步长1截取
            data_seq = data[in_start:in_end, :]  # (15,f)
            X.append(data_seq)
            # 取23点的y值
            if(level==0):
                y.append(data[in_end:out_end,-50:-40]) # (15,f)
            else:
                y.append(data[in_end:out_end, -1])
        in_start += 1  # 滑动步长增加

    return [np.array(X), np.array(y)]

# 划分训练，测试集
def split_dataset(data):
    '''
        该函数切分训练数据和测试数据
    '''
    [x, y] = data
    # 设置随机数种子，保证每次生成的结果都是一样的
    np.random.seed(42)
    # permutation随机生成0-len(data)随机序列
    shuffled_indices = np.random.permutation(len(x))
    # test_ratio为测试集所占的半分比
    test_set_size = int(len(x) * 0.3)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    train_x, test_x = x[train_indices], x[test_indices]
    train_y, test_y = y[train_indices], y[test_indices]

    return [train_x,train_y], [test_x,test_y]

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
    df[df[["相对湿度(%)", "大气压力(hPa)", "风速(m/s)", "风向(°)", "雨量(mm)", "当前太阳辐射强度(W/m²)",
           "累计太阳辐射量(MJ/m²)"]] < 0] = 0
    # 限制雨量，风速数值
    df['雨量(mm)']=np.where(df['雨量(mm)']>40, 40, df['雨量(mm)'])
    df['风速(m/s)'] = np.where(df['风速(m/s)'] > 20, 20, df['风速(m/s)'])
    df1 = pd.read_csv(pathdir[1], header=0)
    df1['时间'] = pd.to_datetime(df1['时间'])
    df1.set_index("时间", inplace=True)
    # 检查时间序列是否连续(使用小时重采样实现)
    df0 = df.resample('H').mean().interpolate()
    df2 = df1.resample('H').mean().interpolate()
    # 保存时间连续的数据
    order = ["空气温度(℃)", "相对湿度(%)", "大气压力(hPa)", "风速(m/s)", "风向(°)", "雨量(mm)","当前太阳辐射强度(W/m²)", "累计太阳辐射量(MJ/m²)"]
    df0 = df0[order]
    df0.to_csv(path_or_buf=pathdir[2], index_label="时间", encoding='utf_8_sig')
    order = ["土壤温度(℃)-地表", '土壤温度(℃)-10', '土壤温度(℃)-20', '土壤温度(℃)-30', '土壤温度(℃)-40',
             '土壤温度(℃)-50', '土壤温度(℃)-60', '土壤温度(℃)-70', '土壤温度(℃)-80', '土壤温度(℃)-90',
             '土壤温度(℃)-100', '水分含量(%)-10', '水分含量(%)-20', '水分含量(%)-30', '水分含量(%)-40',
             '水分含量(%)-50', '水分含量(%)-60', '水分含量(%)-70', '水分含量(%)-80', '水分含量(%)-90',
             '水分含量(%)-100']
    df2 = df2[order]
    df2.to_csv(path_or_buf=pathdir[3], index_label="时间", encoding='utf_8_sig')

    # 计算气象因子的日均值  按时间重采样,计算变量日均值，label=right是指 索引保留右侧的取值，[0:00,23:00]
    df_day = df0.resample('D', label='left').mean().reset_index()
    # 获取当天的气象，以及向后滑动15天的气象
    df0_forw15 = df_day.shift(-15).add_suffix("_t+15").iloc[:, 1:] # 舍弃时间列
    meteo = pd.concat([df_day, df0_forw15], axis=1)
    # 保存t,t+15的气象数据数据
    meteo.to_csv(os.path.join(rootpath,"Meteo"+str(point)+".csv"),encoding='utf_8_sig',index=False)

    # 取出每天23:00的土壤温湿度数据  x指 行索引，其中，表头为0
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

    # 水分数据
    df_SM = df_23h[["时间",'水分含量(%)-10', '水分含量(%)-20', '水分含量(%)-30', '水分含量(%)-40',
                     '水分含量(%)-50', '水分含量(%)-60', '水分含量(%)-70', '水分含量(%)-80', '水分含量(%)-90', '水分含量(%)-100']]
    # 查看缺失值
    print(df_SM.isna().sum())
    #  添加土壤容重数据
    df_SM['bd10'], df_SM['bd20'], df_SM['bd30'], df_SM['bd40'], df_SM['bd50'], df_SM[
        'bd60'], df_SM['bd70'], df_SM['bd80'], df_SM['bd90'], df_SM['bd100'] = bulk_density
    # 加入土壤组成数据
    df_SM['sand10'], df_SM['sand20'], df_SM['sand30'], df_SM['sand40'], df_SM['sand50'], \
    df_SM['sand60'], df_SM['sand70'], df_SM['sand80'], df_SM['sand90'], df_SM['sand100'], \
    df_SM['silt10'], df_SM['silt20'], df_SM['silt30'], df_SM['silt40'], df_SM['silt50'], \
    df_SM['silt60'], df_SM['silt70'], df_SM['silt80'], df_SM['silt90'], df_SM['silt100'], \
    df_SM['clay10'], df_SM['clay20'], df_SM['clay30'], df_SM['clay40'], df_SM['clay50'], \
    df_SM['clay60'], df_SM['clay70'], df_SM['clay80'], df_SM['clay90'], df_SM['clay100'] = soilComp

    # 另存为csv
    df_SM.to_csv(os.path.join(rootpath,"SM_23h"+str(point)+".csv"),encoding='utf_8_sig',index=False)


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


# 读取各个变量并拼接
def concatVariable(date_path,meto_ave_path,SM_path,ST_23h_path,savepath):
    # 各个csv有标题行，首列为时间列
    date = pd.read_csv(date_path, header=0)
    meto = pd.read_csv(meto_ave_path, header=0)
    ST_23h = pd.read_csv(ST_23h_path, header=0)
    SM = pd.read_csv(SM_path, header=0)
    variable = pd.concat([date,meto.iloc[:,1:],ST_23h.iloc[:,1:],SM.iloc[:,11:],SM.iloc[:,1:11]],axis=1)
    # variable = pd.concat([meto, SM.iloc[:, 11:], SM.iloc[:, 1:11]], axis=1)
    variable.to_csv(savepath, encoding='utf_8_sig',index=False)
    return


if __name__ == '__main__':

    # 先处理预测点3的数据
    # interpolate('./data/rawData/SMST3.csv','./data/rawData_csv/SMST3_new.csv')
    #
    rootpath = "./lstmData/inputData"
    pathdir1 = ['./data/rawData/Meteorology1.csv', './data/rawData/STSM1.csv',
                './lstmData/checkData/Meteorology1.csv', './lstmData/checkData/STSM1.csv', './lstmData/inputData/variable1.csv']
    pathdir2 = ['./data/rawData/Meteorology2.csv', './data/rawData/STSM2.csv',
                './lstmData/checkData/Meteorology2.csv', './lstmData/checkData/STSM2.csv', './lstmData/inputData/variable2.csv']
    # pathdir3 = ['./data/rawData/Meteorology3.csv', './data/rawData/STSM3_new.csv',
    #             './lstmData/checkData/Meteorology3.csv', './lstmData/checkData/STSM3.csv', './lstmData/inputData/variable3.csv']
    pathdir4 = ['./data/rawData/Meteorology4_new.csv', './data/rawData/STSM4_new.csv',
                './lstmData/checkData/Meteorology4.csv', './lstmData/checkData/STSM4.csv', './lstmData/inputData/variable4.csv']
    pathdir5_1 = ['./data/rawData/Meteorology5_new1.csv', './data/rawData/STSM5_new1.csv',
                  './lstmData/checkData/Meteorology5_1.csv', './lstmData/checkData/STSM5_1.csv','./lstmData/inputData/variable5_1.csv']
    pathdir5_2 = ['./data/rawData/Meteorology5_new2.csv', './data/rawData/STSM5_new2.csv',
                  './lstmData/checkData/Meteorology5_2.csv', './lstmData/checkData/STSM5_2.csv','./lstmData/inputData/variable5_2.csv']
    pathdir5_3 = ['./data/rawData/Meteorology5_new3.csv', './data/rawData/STSM5_new3.csv',
                  './lstmData/checkData/Meteorology5_3.csv', './lstmData/checkData/STSM5_3.csv','./lstmData/inputData/variable5_3.csv']
    pathdir6_1 = ['./data/rawData/Meteorology6_new1.csv', './data/rawData/STSM6_new1.csv',
                  './lstmData/checkData/Meteorology6_1.csv', './lstmData/checkData/STSM6_1.csv','./lstmData/inputData/variable6_1.csv']
    pathdir6_2 = ['./data/rawData/Meteorology6_new2.csv', './data/rawData/STSM6_new2.csv',
                  './lstmData/checkData/Meteorology6_2.csv', './lstmData/checkData/STSM6_2.csv','./lstmData/inputData/variable6_2.csv']
    # 读取土壤容重数据
    bddf = pd.read_csv("./data/rawData/bulk_density.csv",header=0)
    # 读取土壤组成数据
    soilComp = pd.read_csv('./data/rawData/soilComponent.csv',header=0)
    # readrawData(1,rootpath,pathdir1, bddf.loc[0],soilComp.loc[0])
    # readrawData(2,rootpath,pathdir2, bddf.loc[1],soilComp.loc[1])
    # # readrawData(3,rootpath,pathdir3, bddf.loc[2],soilComp.loc[2])
    # readrawData(4,rootpath,pathdir4, bddf.loc[3],soilComp.loc[3])
    # readrawData(5_1,rootpath,pathdir5_1, bddf.loc[4],soilComp.loc[4])
    # readrawData(5_2, rootpath, pathdir5_2, bddf.loc[4], soilComp.loc[4])
    # readrawData(5_3, rootpath, pathdir5_3, bddf.loc[4], soilComp.loc[4])
    # readrawData(6_1,rootpath,pathdir6_1,bddf.loc[5],soilComp.loc[5])
    # readrawData(6_2, rootpath, pathdir6_2, bddf.loc[5], soilComp.loc[5])


    # 读取各个变量并拼接
    date_path = ["./lstmData/inputData/date1.csv","./lstmData/inputData/date2.csv","./lstmData/inputData/date3.csv",
    "./lstmData/inputData/date4.csv","./lstmData/inputData/date51.csv","./lstmData/inputData/date52.csv","./lstmData/inputData/date53.csv","./lstmData/inputData/date61.csv","./lstmData/inputData/date62.csv"]
    meto_path = ["./lstmData/inputData/Meteo1.csv","./lstmData/inputData/Meteo2.csv","./lstmData/inputData/Meteo3.csv",
    "./lstmData/inputData/Meteo4.csv","./lstmData/inputData/Meteo51.csv","./lstmData/inputData/Meteo52.csv","./lstmData/inputData/Meteo53.csv","./lstmData/inputData/Meteo61.csv","./lstmData/inputData/Meteo62.csv"]
    SM_path = [".\lstmData\inputData\SM_23h1.csv",".\lstmData\inputData\SM_23h2.csv",".\lstmData\inputData\SM_23h3.csv",
                   ".\lstmData\inputData\SM_23h4.csv",".\lstmData\inputData\SM_23h51.csv",".\lstmData\inputData\SM_23h52.csv",".\lstmData\inputData\SM_23h53.csv",".\lstmData\inputData\SM_23h61.csv",".\lstmData\inputData\SM_23h62.csv",]
    ST_23h_path = [".\lstmData\inputData\ST_23h1.csv",".\lstmData\inputData\ST_23h2.csv",".\lstmData\inputData\ST_23h3.csv",
                   ".\lstmData\inputData\ST_23h4.csv",".\lstmData\inputData\ST_23h51.csv",".\lstmData\inputData\ST_23h52.csv",".\lstmData\inputData\ST_23h53.csv",".\lstmData\inputData\ST_23h61.csv",".\lstmData\inputData\ST_23h62.csv"]
    savepath=["./lstmData/inputData/variable_st1.csv","./lstmData/inputData/variable_st2.csv","./lstmData/inputData/variable_st3.csv",
              "./lstmData/inputData/variable_st4.csv","./lstmData/inputData/variable_st51.csv","./lstmData/inputData/variable_st52.csv","./lstmData/inputData/variable_st53.csv","./lstmData/inputData/variable_st61.csv","./lstmData/inputData/variable_st62.csv"]
    bulk_density = "./data/rawData/bulk_density.csv"
    soilComponent = "./data/rawData/soilComponent.csv"
    # # 读取变量数据，拼接在一起获得输入输出变量数据
    # concatVariable(date_path[0], meto_path[0], SM_path[0], ST_23h_path[0], savepath[0])
    # concatVariable(date_path[1], meto_path[1], SM_path[1], ST_23h_path[1], savepath[1])
    # # concatVariable(date_path[2], meto_path[2], SM_path[2], ST_23h_path[2], savepath[2])
    # concatVariable(date_path[3], meto_path[3], SM_path[3], ST_23h_path[3], savepath[3])
    # concatVariable(date_path[4], meto_path[4], SM_path[4], ST_23h_path[4], savepath[4])
    # concatVariable(date_path[5], meto_path[5], SM_path[5], ST_23h_path[5], savepath[5])
    # concatVariable(date_path[6], meto_path[6], SM_path[6], ST_23h_path[6], savepath[6])
    # concatVariable(date_path[7], meto_path[7], SM_path[7], ST_23h_path[7], savepath[7])
    # concatVariable(date_path[8], meto_path[8], SM_path[8], ST_23h_path[8], savepath[8])


    # 读取6个预测点的变量数据  read_csv 不需要关闭
    dataset1 = pd.read_csv("./lstmData/inputData/variable_st1.csv", header=0)
    dataset2 = pd.read_csv("./lstmData/inputData/variable_st2.csv", header=0)
    dataset3 = pd.read_csv("./lstmData/inputData/variable_st3.csv", header=0)
    dataset4 = pd.read_csv("./lstmData/inputData/variable_st4.csv", header=0)
    dataset5_1 = pd.read_csv("./lstmData/inputData/variable_st51.csv", header=0)
    dataset5_2 = pd.read_csv("./lstmData/inputData/variable_st52.csv", header=0)
    dataset5_3 = pd.read_csv("./lstmData/inputData/variable_st53.csv", header=0)
    dataset6_1 = pd.read_csv("./lstmData/inputData/variable_st61.csv", header=0)
    dataset6_2 = pd.read_csv("./lstmData/inputData/variable_st62.csv", header=0)


    sliding_window_width = 15
    output_lenght=15
    input_sequence_start = 0

    # 先滑动窗口，再拆分数据集 （错误！！先拆分数据集再滑动，保证训练，测试集时间不重叠）
    [data_x1, data_y1] = sliding_window(dataset1, sliding_window_width,output_lenght, in_start=input_sequence_start,a=0,b=591,level=1)
    [data_x2, data_y2] = sliding_window(dataset2, sliding_window_width, output_lenght,in_start=input_sequence_start,a=0,b=591,level=1)
    [data_x3, data_y3] = sliding_window(dataset3, sliding_window_width, output_lenght,in_start=input_sequence_start,a=0,b=591,level=2)
    [data_x4, data_y4] = sliding_window(dataset4, sliding_window_width, output_lenght,in_start=input_sequence_start,a=0,b=463,level=1)
    [data_x5_1, data_y5_1] = sliding_window(dataset5_1, sliding_window_width, output_lenght,in_start=input_sequence_start,a=0,b=203,level=1)
    [data_x5_2, data_y5_2] = sliding_window(dataset5_2, sliding_window_width, output_lenght,in_start=input_sequence_start,a=0,b=137,level=1)
    [data_x5_3, data_y5_3] = sliding_window(dataset5_3, sliding_window_width, output_lenght,in_start=input_sequence_start,a=0,b=40,level=1)
    [data_x6_1, data_y6_1] = sliding_window(dataset6_1, sliding_window_width,output_lenght, in_start=input_sequence_start,a=0,b=233,level=1)
    [data_x6_2, data_y6_2] = sliding_window(dataset6_2, sliding_window_width, output_lenght, in_start=input_sequence_start,a=0, b=353, level=1)

    # 先拼接所有预测点的数据, level1 去掉预测点6的数据
    data_x = np.concatenate((data_x1, data_x2,data_x4,data_x5_1,data_x5_2,data_x5_3,), axis=0) # data_x6_1,data_x6_2
    data_y = np.concatenate((data_y1, data_y2,data_y4,data_y5_1,data_y5_2,data_y5_3,), axis=0) # data_y6_1,data_y6_2
    # 对每个预测点划分测试集和训练集
    train, test = split_dataset([data_x, data_y])
    train, scaler_trainy= scalerVariable(train[0],train[1],0,0)
    epochs_num = 1000
    batch_size_set = 50
    verbose_set = 2  # 显示每个epoch的记录


    # todo 预测未来15天的下一层，作为约束加入变量
    # ---------------------------------------------使用训练好的模型预测下一层，并作为变量加入------------------------------------------
    # yyhat = pd.read_csv("./lstmData/nextLayer/trainY8.csv",header=None)
    # train[0] = np.append(train[0], np.array(yyhat).reshape(-1,15,1), axis=2)
    # 第l层添加第l+1层预测结果
    # model1 = load_model('./save_model/lstm_encoder_decoder_level3/LSTM.835-0.0003-0.0178.h5')
    # [ddata_x1, ddata_y1] = sliding_window(dataset1, sliding_window_width, output_lenght, in_start=input_sequence_start,a=0, b=591, level=3)
    # [ddata_x2, ddata_y2] = sliding_window(dataset2, sliding_window_width, output_lenght, in_start=input_sequence_start,a=0, b=591, level=3)
    # [ddata_x3, ddata_y3] = sliding_window(dataset3, sliding_window_width, output_lenght, in_start=input_sequence_start,a=0, b=591, level=3)
    # [ddata_x4, ddata_y4] = sliding_window(dataset4, sliding_window_width, output_lenght, in_start=input_sequence_start,a=0, b=591, level=3)
    # [ddata_x5_1, ddata_y5_1] = sliding_window(dataset5_1, sliding_window_width, output_lenght,in_start=input_sequence_start, a=0, b=203, level=3)
    # [ddata_x5_2, ddata_y5_2] = sliding_window(dataset5_2, sliding_window_width, output_lenght,in_start=input_sequence_start, a=0, b=137, level=3)
    # [ddata_x5_3, ddata_y5_3] = sliding_window(dataset5_3, sliding_window_width, output_lenght,in_start=input_sequence_start, a=0, b=40, level=3)
    # [ddata_x6_1, ddata_y6_1] = sliding_window(dataset6_1, sliding_window_width, output_lenght,in_start=input_sequence_start, a=0, b=233, level=3)
    # [ddata_x6_2, ddata_y6_2] = sliding_window(dataset6_2, sliding_window_width, output_lenght,in_start=input_sequence_start, a=0, b=353, level=3)
    # ddata_x = np.concatenate((ddata_x1, ddata_x2, ddata_x3, ddata_x4, ddata_x5_1, ddata_x5_2, ddata_x5_3,ddata_x6_1,ddata_x6_2),axis=0)  # ddata_x6_1,ddata_x6_2
    # ddata_y = np.concatenate((ddata_y1, ddata_y2, ddata_y3, ddata_y4, ddata_y5_1, ddata_y5_2, ddata_y5_3,ddata_y6_1,ddata_y6_2),axis=0)  # ddata_y6_1,ddata_y6_2
    # dtrain, dtest = split_dataset([ddata_x, ddata_y])
    # dtrain, dscaler_trainy = scalerVariable(dtrain[0], dtrain[1], 0, 0)
    # yyhat = model1.predict(dtrain[0], verbose=0)
    # train[0]=np.append(train[0], yyhat, axis = 2)
    # # l+1层的测试集
    # [dtest_x, dtest_y] = dtest
    # dtest_x = dtest_x.reshape(dtest_x.shape[0], -1)
    # dtest_y = dtest_y.reshape(dtest_y.shape[0], -1)
    # min_max = pd.read_csv('./lstmData/min_max_level3_st.csv', header=None)
    # dx_max = np.array(min_max.iloc[:, 0])
    # dx_min = np.array(min_max.iloc[:, 1])
    # dy_max = np.array(min_max.iloc[:, 2])
    # dy_min = np.array(min_max.iloc[:, 3])
    # dX = (dtest_x - dx_min) / (dx_max - dx_min)
    # dY = (dtest_y - dy_min[0:15]) / (dy_max[0:15] - dy_min[0:15])
    # dX = dX.reshape(dtest_x.shape[0], sliding_window_width, -1)  # [n,15,f]
    # yyhat = model1.predict(dX, verbose=0)  # l+1层的测试集结果
    # ----------------------------------------------------------------------------------------------------------


    # verbose = 0 为不在标准输出流输出日志信息，verbose = 1 为输出进度条记录，verbose = 2 为每个epoch输出一行记录
    # 输入 [n,f,t] 输出 [n,10,15]
    # lstm_model(train, sliding_window_width, input_sequence_start, verbose_set=verbose_set, epochs_num=epochs_num, batch_size_set=batch_size_set)
    # encoder_decoder_lstm(train, sliding_window_width, input_sequence_start, verbose_set=verbose_set, epochs_num=epochs_num,batch_size_set=batch_size_set)
    # biLSTM_model(train, sliding_window_width, input_sequence_start, verbose_set=verbose_set, epochs_num=epochs_num,batch_size_set=batch_size_set)
    # tcn_model(train, sliding_window_width, input_sequence_start, verbose_set=verbose_set, epochs_num=500,batch_size_set=batch_size_set)

    [test_x,test_y] = test
    # 归一化
    test_x = test_x.reshape(test_x.shape[0], -1)
    test_y = test_y.reshape(test_y.shape[0], -1)
    min_max = pd.read_csv('./lstmData/min_max_level1_st.csv', header=None)
    x_max = np.array(min_max.iloc[:,0])
    x_min = np.array(min_max.iloc[:, 1])
    y_max = np.array(min_max.iloc[:,2])
    y_min = np.array(min_max.iloc[:,3])

    X = (test_x - x_min) / (x_max - x_min)
    Y = (test_y - y_min[0:15]) / (y_max[0:15] - y_min[0:15])
    X = X.reshape(test_x.shape[0], sliding_window_width, -1)  # [n,15,f]
    model = load_model('./save_model/lstm_encoder_decoder_level1/LSTM.565-0.0017-0.0416.h5')
    # ----------------------------------------------------------------------------------------------------------
    # 为测试集添加l+1层的变量
    # yyhat=pd.read_csv("./lstmData/nextLayer/testY8.csv",header=None)
    # X = np.append(X, np.array(yyhat).reshape(-1,15,1), axis=2)
    # 第l层的训练集结果
    # trainY = model.predict(train[0], verbose=0)
    # pd.DataFrame(trainY.reshape(-1, 15)).to_csv("./lstmData/nextLayer/trainY7.csv", index=False, header=None)
    # ----------------------------------------------------------------------------------------------------------
    # 第l层的测试集结果
    yhat = model.predict(X, verbose=0)
    yhat = yhat.reshape(yhat.shape[0],15)
    # pd.DataFrame(yhat).to_csv("./lstmData/nextLayer/testY7.csv", index=False, header=None)

    predicts = yhat*(y_max[0:15] - y_min[0:15])+y_min[0:15]
    predicts = predicts.reshape(test_x.shape[0],15)
    test_y = test_y.reshape(test_y.shape[0],15)
    # 每天预测误差
    R2 = []
    scores = list()
    for i in range(test_y.shape[1]):
        mse = skm.mean_squared_error(test_y[:,i], predicts[:,i])
        rmse = math.sqrt(mse)
        scores.append(rmse)
        R2.append(skm.r2_score(test_y[:,i], predicts[:,i]))
    print('\n每个深度的RMSE:\n', scores)
    print('\n每个深度的R2:\n', R2)



print("End!!")