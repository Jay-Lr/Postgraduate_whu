"""
    补充连续时间序列
    计算各个变量的日均值
    提取23：00的土壤墒情数据
    以步长15滑动窗口，建立数据集
    建立LSTM模型，输入15步，输出15步，一次预测未来15天的10个变量
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
from sklearn.metrics import r2_score
from tensorflow.keras.models import load_model
from tensorflow import keras
# 中文字体显示
plt.rcParams['font.family']='Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False



# 绘制各个变量的趋势图变化
def plot_features(dataset):
    # 创建一个包含八个子图的图像，每个子图对应一个变量
    dataset = dataset.iloc[:,0:9]
    plt.figure(figsize=(16, 16), dpi=300)
    for i in range(1,len(dataset.columns)): # 除去时间列
        plt.subplot(len(dataset.columns), 1, i)
        feature_name = dataset.columns[i]
        plt.plot(dataset[feature_name],color = 'k',)
        plt.rcParams.update({'font.size': 12})
        plt.title(feature_name,size=16)
        plt.grid(linestyle='--', alpha=0.5)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

    plt.tight_layout()
    # plt.show()
    plt.savefig(dataset.columns[1] + '.tif',dpi=300)


def lstm_model(train_x,train_y, sw_width, in_start=0, verbose_set=0, epochs_num=20, batch_size_set=4):
    '''
        该函数定义 LSTM 模型
    '''
    if(train_y.ndim==3):
        train_y_old = train_y
        train_y = train_Y.reshape(train_y.shape[0],-1)
    # 使用model的callbacks函数
    checkpoint = keras.callbacks.ModelCheckpoint(filepath='./save_model/LSTM.{epoch:03d}-{val_loss:.4f}.h5',
                                 monitor='val_loss',  # 验证集损失函数
                                 verbose=1,
                                 save_best_only=False)  # 若设置为True，则只保存最好的模型
    # lr_reducer = keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),  # 缩放学习率的值，学习率将以lr = lr*factor的形式被减少
    #                                cooldown=0,
    #                                patience=5,
    #                                min_lr=0.5e-6)

    # callbacks = [checkpoint, lr_reducer]
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

    model = Sequential()
    model.add(LSTM(100, activation='relu',return_sequences=True,  # return_sequences=True,表示是否多对多数据传递到下一个神经层网络
                   input_shape=(n_timesteps, n_features)))
    model.add(LSTM(100, activation='relu'))
    # model.add(LSTM(100, activation='relu'))
    # model.add(LSTM(100, activation='relu'))
    model.add(Dense(n_outputs))

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    # 使用训练集中的10%进行验证，观察验证集的效果选择最佳模型
    history = model.fit(train_x, train_y,
              epochs=epochs_num, batch_size=batch_size_set, verbose=verbose_set,validation_split=0.1, shuffle=True,
                        callbacks=[checkpoint])
    print('\ntrain_acc:%s' % np.mean(history.history['accuracy']), '\ntrain_loss:%s' % np.mean(history.history['loss']))
    # 恢复train_y的维度
    # train_yy = train_y.reshape(train_y,15,-1)
    # TODO 损失函数绘图
    # 创建一个绘图窗口
    plt.figure()
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')  # 'bo'为画蓝色圆点，不连线
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()  # 绘制图例，默认在右上角

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    return model

# 编码-解码LSTM
def encoder_decoder_lstm(self):
    model = Sequential()
    model.add(LSTM(200, activation='relu',
                   input_shape=(self.sw_width, self.features)))
    model.add(RepeatVector(self.pred_length))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(self.features)))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    print(model.summary())

    history = model.fit(self.X, self.y, epochs=self.epochs_num, verbose=self.verbose_set)
    print('\ntrain_acc:%s' % np.mean(history.history['accuracy']),
          '\ntrain_loss:%s' % np.mean(history.history['loss']))

# 计算真值与预测值的均方差
def evaluate_forecasts(actual, predicted):
    '''
        scores 每天 所有样本 的均方差
        score 每个样本 每天预测误差 的均方之和/样本/15天，平均每个样本每个步长的误差
    '''
    scores = list()
    R2=[]
    # [n,15,10]
    for i in range(actual.shape[1]):
        mse = skm.mean_squared_error(actual[:, i], predicted[:, i])
        rmse = math.sqrt(mse)
        scores.append(rmse)
        # TODO　计算R2 多输出
        R2.append(r2_score(actual[:,i], predicted[:,i]))


    s = 0  # 计算总的 RMSE
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = math.sqrt(s / (actual.shape[0] * actual.shape[1]))
    print('actual.shape[0]:{}, actual.shape[1]:{}'.format(actual.shape[0], actual.shape[1]))


    return score, scores


def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s\n' % (name, score, s_scores))


def model_plot(score, scores, days, name):
    '''
    该函数实现绘制RMSE曲线图
    '''
    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(days, scores, marker='o', label=name)
    plt.grid(linestyle='--', alpha=0.5)
    plt.ylabel(r'$RMSE$', size=15)
    plt.title('LSTM 模型预测结果', size=18)
    plt.legend()
    plt.show()

# 划分训练，测试集
def split_dataset(data):
    '''
        该函数切分训练数据和测试数据
    '''
    # 6：4划分
    # 测试集数据 60个变量
    train, test = data.iloc[0:-184,2:62], data.iloc[-184:-15,2:62]
    # train = np.array(np.split(train, len(train) / 15/24))  # 将数据划分为按15天为单位的数据
    # test = np.array(np.split(test, len(test) / 15/24))
    return np.array(train), np.array(test)


# 滑动窗口
def sliding_window(train, sw_width=30, n_out=15, in_start=0):
    '''
    该函数实现窗口宽度为45(输入30，输出15)、滑动步长为1天的滑动窗口截取序列数据
    '''
    data = train.reshape((train.shape[0], train.shape[1]))  #  二维 n,f
    X, y = [], []

    for _ in range(len(data)):
        in_end = in_start + sw_width
        out_end = in_end + n_out

        # 保证截取样本完整，最大元素索引不超过原序列索引，则截取数据；否则丢弃该样本
        if out_end < len(data):
            # 训练数据以滑动步长1截取
            train_seq = data[in_start:in_end, :]  # (15,f)
            X.append(train_seq)
            # 取23点的y值
            y.append(data[in_end:out_end,-10:]) # (15,f)
        in_start += 1  # 滑动步长增加

    return np.array(X), np.array(y)

# 变换y的形状
def reshape_y(y):
    sample_num = y.shape[0]
    step_length = y.shape[1]
    feature_num = y.shape[2]
    y1 = y.reshape(sample_num,step_length*feature_num)

    return y1

# 恢复y的形状
def reshape_y_back(y,step_length):
    sample_num = y.shape[0]
    y1=y.reshape(sample_num,step_length,-1)
    return y1


def main_run(train,test,scaler_trainy,scaler_testy, sw_width, name, in_start, verbose, epochs, batch_size):
    '''
    主函数：数据处理、模型训练流程
    '''
    train_x,train_y = train[0],train[1]
    # TODO train_y维度转换，转为[n,15*10]
    train_y = reshape_y(train_y)

    # 训练模型
    # verbose = 0 为不在标准输出流输出日志信息，verbose = 1 为输出进度条记录，verbose = 2 为每个epoch输出一行记录
    lstm_model(train_x,train_y, sw_width, in_start, verbose_set=verbose, epochs_num= epochs, batch_size_set=batch_size)

def forecast(model, test):
    test_x, test_y = test[0], test[1]
    step_length = test_y.shape[1]
    predictions = model.predict(test_x, verbose=0)
    predictions = np.array(predictions)
    # 评价
    test_y = reshape_y(test_y)
    score, scores = evaluate_forecasts(test_y, predictions)
    # 打印分数
    summarize_scores(name, score, scores)
    # 预测值变换为原始维度
    predictions =reshape_y_back( predictions,step_length)
    return predictions


# 处理原始数据
def readrawData(pathdir,bulk_density):
    """
        预测点1,2,3,4，分辨率为小时级，
        预测点 5,6，分辨率为分钟级，
        使用小时采样保证时间连续且插值
    """
    # 首行为列名，第一列时间列为索引列 dataframe
    df = pd.read_csv(pathdir[0],  header=0) # encoding="gbk"
    df['时间'] = pd.to_datetime(df['时间'])
    df.set_index("时间", inplace=True)
    df1 = pd.read_csv(pathdir[1], header=0 ) # encoding="gbk"
    df1['时间'] = pd.to_datetime(df1['时间'])
    df1.set_index("时间", inplace=True)
    # 检查时间序列是否连续(使用小时重采样实现)
    df0 = df.resample('H').mean().interpolate()
    df2 = df1.resample('H').mean().interpolate()
    # df0.to_csv(path_or_buf=pathdir[2], index_label="时间", encoding='utf_8_sig')
    # df2.to_csv(path_or_buf=pathdir[3],index_label="时间",encoding='utf_8_sig')

    # 取出每天23:00的数据  x指 行索引，其中，表头为0
    df_23h = pd.read_csv(pathdir[3], header=0,skiprows=lambda x: x > 0 and x % 24 != 0)


    # 为避免列名重复
    df_23h = df_23h.add_suffix("_y")
    # 两个数据拼接合并,为重复列名添加后缀
    df_new= df0.join(df2, rsuffix='_right')
    # 转换时间列，并设为索引
    # df_new['时间'] = pd.to_datetime(df_new['时间'])
    # df_new.set_index("时间", inplace=True)
    print(df_new.shape)
    print(df_new.head(10))
    # TODO 加入土壤质地数据

    # 按时间重采样,计算变量日均值，label=right是指 索引保留右侧的取值，[0:00,23:00]
    df_day = df_new.resample('D',label= 'left').mean().reset_index()
    print(df_day.head(10))

    # 查看缺失值
    print(df_day.isna().sum())  # 无缺失值
    # 为方便表示，重命名各列
    # df.rename(columns={'相对湿度(%)':'Humi', '大气压力(hPa)':'Press', '风速(m/s)':'windV','风向(°)':'windD','雨量(mm)':'Preci','当前太阳辐射强度(W/m²)':'nSRI',
    #                    '累计太阳辐射量(MJ/m²)':'cSRI','空气温度(℃)':'Tem'}, inplace=True)
    # 绘制变量变化趋势
    plot_features(df_day)
    # 23点的变量值
    df_day_all = df_day.join(df_23h)
    #  土壤容重数据
    df_day_all['bd10'],df_day_all['bd20'],df_day_all['bd30'],df_day_all['bd40'],df_day_all['bd50'],df_day_all['bd60'],df_day_all['bd70'],df_day_all['bd80'],df_day_all['bd90'],df_day_all['bd100']=bulk_density

    print(df_day_all.columns.values)
    # 确保列名顺序一致 '土壤温度(℃)-120','土壤温度(℃)-150','水分含量(%)-120','水分含量(%)-150',
    #            '土壤温度(℃)-120_y', '土壤温度(℃)-150_y', '水分含量(%)-120_y', '水分含量(%)-150_y' 是预测点3 使用的
    order=['时间','时间_y','相对湿度(%)','大气压力(hPa)','风速(m/s)','风向(°)','雨量(mm)','当前太阳辐射强度(W/m²)','累计太阳辐射量(MJ/m²)'	,'空气温度(℃)','土壤温度(℃)-地表','土壤温度(℃)-10','土壤温度(℃)-20','土壤温度(℃)-30',
           '土壤温度(℃)-40','土壤温度(℃)-50','土壤温度(℃)-60','土壤温度(℃)-70','土壤温度(℃)-80','土壤温度(℃)-90','土壤温度(℃)-100','水分含量(%)-10'	,'水分含量(%)-20','水分含量(%)-30'	,'水分含量(%)-40',
           '水分含量(%)-50','水分含量(%)-60','水分含量(%)-70','水分含量(%)-80','水分含量(%)-90','水分含量(%)-100','bd10','bd20','bd30','bd40','bd50','bd60','bd70','bd80','bd90','bd100','土壤温度(℃)-地表_y','土壤温度(℃)-10_y',	'土壤温度(℃)-20_y','土壤温度(℃)-30_y',
           '土壤温度(℃)-40_y','土壤温度(℃)-50_y',	'土壤温度(℃)-60_y','土壤温度(℃)-70_y','土壤温度(℃)-80_y','土壤温度(℃)-90_y','土壤温度(℃)-100_y','水分含量(%)-10_y','水分含量(%)-20_y','水分含量(%)-30_y',
           '水分含量(%)-40_y','水分含量(%)-50_y','水分含量(%)-60_y','水分含量(%)-70_y','水分含量(%)-80_y','水分含量(%)-90_y','水分含量(%)-100_y'
            ]
    df_day_all = df_day_all[order]
    # 另存为csv
    df_day_all.to_csv(path_or_buf=pathdir[4],index_label="时间",encoding='utf-8_sig',index=False)
    # 归一化，需要保存预测变量的最大最小值


# 预测点3缺少70，90cm的土壤温湿度，多项式拟合土壤深度和温湿度，插值
def interpolate(path,savepath):
    df = pd.read_csv(path,header=0)
    # 温度数据 考虑到地表温湿度受外界条件影响较大，只是有地下数据
    x=[10,20,30,40,50,60,80,100,120,150]
    x0=[70,90]
    ST = df.iloc[:,2:12]
    SM = df.iloc[:,12:]
    ST_y = np.empty(shape=[0, 2], dtype=float)
    SM_y = np.empty(shape=[0, 2], dtype=float)
    R_square=[]
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
        SM_y = np.append(SM_y,[SMiy3], axis=0)  # 添加整行元素，axis=1添加整列元素
        # R_square.append([r2_score(ST.loc[i],spi.splev(x, STipo3).tolist()), r2_score(SM.loc[i], spi.splev(x0, SMipo3).tolist())])
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        # ax1.plot(x, ST.loc[i], label='原数值')
        # ax1.plot(x0, STiy3, 'r.', label='插值点')
        # ax2.plot(x, SM.loc[i], label='原数值')
        # ax2.plot(x0, SMiy3, 'b.', label='插值点')

    SM_y = np.around(SM_y,  decimals=4)
    ST_y = np.around(ST_y, decimals=4)
    # 为df插入两列
    dfSM_y = pd.DataFrame(SM_y)
    dfST_y = pd.DataFrame(ST_y)
    df['土壤温度(℃)-70'],df['土壤温度(℃)-90'] = dfST_y.iloc[:,0], dfST_y.iloc[:,1]
    df['水分含量(%)-70'], df['水分含量(%)-90'] = dfSM_y.iloc[:,0],dfSM_y.iloc[:,1]
    df.to_csv(path_or_buf=savepath, encoding='utf-8_sig', index=False)
    return

# TODO 归一化
def scalerVariable(dataset):
    df = dataset
    df = (df - df.min()) / (df.max() - df.min())
    # 暂时不对y（土壤湿度）做归一化，划分过数据集，拼接后再对y做归一化


if __name__ == '__main__':
    """
           整理拼接全部数据 气象+墒情

    """
    # 先处理预测点3的数据
    # interpolate('./data/rawData_csv/SMST3.csv','./data/rawData_csv/SMST3_new.csv')

    pathdir1 = ['./data/rawData_csv/Meteorology1.csv', './data/rawData_csv/SMST1.csv',
                './data/checkData/Meteorology1.csv', './data/checkData/SMST1.csv', './data/inputData/variable1.csv']
    pathdir2 = ['./data/rawData_csv/Meteorology2.csv', './data/rawData_csv/SMST2.csv',
                './data/checkData/Meteorology2.csv', './data/checkData/SMST2.csv', './data/inputData/variable2.csv']
    pathdir3 = ['./data/rawData_csv/Meteorology3.csv', './data/rawData_csv/SMST3_new.csv',
                './data/checkData/Meteorology3.csv', './data/checkData/SMST3.csv', './data/inputData/variable3.csv']
    pathdir4 = ['./data/rawData_csv/Meteorology4.csv', './data/rawData_csv/SMST4.csv',
                './data/checkData/Meteorology4.csv', './data/checkData/SMST4.csv', './data/inputData/variable4.csv']
    pathdir5 = ['./data/rawData_csv/Meteorology5.csv', './data/rawData_csv/SMST5.csv',
                './data/checkData/Meteorology5.csv', './data/checkData/SMST5.csv', './data/inputData/variable5.csv']
    pathdir6=['./data/rawData_csv/Meteorology6.csv','./data/rawData_csv/SMST6.csv',
              './data/checkData/Meteorology6.csv','./data/checkData/SMST6.csv','./data/inputData/variable6.csv']
    # 读取土壤容重数据
    bddf = pd.read_csv("./data/rawData_csv/bulk_density.csv",header=0)
    readrawData(pathdir1, bddf.loc[0])
    readrawData(pathdir2, bddf.loc[1])
    readrawData(pathdir3, bddf.loc[2])
    readrawData(pathdir4, bddf.loc[3])
    readrawData(pathdir5, bddf.loc[4])
    readrawData(pathdir6,bddf.loc[5])

    # 读取6个预测点的变量数据
    dataset1 = pd.read_csv("./data/inputData/variable1.csv",header=0)
    dataset2 = pd.read_csv("./data/inputData/variable2.csv", header=0)
    dataset3 = pd.read_csv("./data/inputData/variable3.csv", header=0)
    dataset4 = pd.read_csv("./data/inputData/variable4.csv", header=0)
    dataset5 = pd.read_csv("./data/inputData/variable5.csv", header=0)
    dataset6 = pd.read_csv("./data/inputData/variable6.csv", header=0)

    # 对每个文件归一化



    sliding_window_width = 15
    input_sequence_start = 0

    # 对每个预测点划分测试集和训练集
    train1, test1 = split_dataset(dataset1)
    train2, test2 = split_dataset(dataset2)
    train3, test3 = split_dataset(dataset3)
    train4, test4 = split_dataset(dataset4)
    train5, test5 = split_dataset(dataset5)
    train6, test6 = split_dataset(dataset6)


    # 滑动窗口切片 x=>[n,15,60],y=>[n,15,10]
    train_x1, train_y1 = sliding_window(train1,  sliding_window_width, in_start=input_sequence_start)
    train_x2, train_y2 = sliding_window(train2, sliding_window_width, in_start=input_sequence_start)
    train_x3, train_y3 = sliding_window(train3, sliding_window_width, in_start=input_sequence_start)
    train_x4, train_y4 = sliding_window(train4, sliding_window_width, in_start=input_sequence_start)
    train_x5, train_y5 = sliding_window(train5, sliding_window_width, in_start=input_sequence_start)
    train_x6, train_y6 = sliding_window(train6, sliding_window_width, in_start=input_sequence_start)

    test_x1, test_y1 = sliding_window(test1,  sliding_window_width, in_start=input_sequence_start)
    test_x2, test_y2 = sliding_window(test2, sliding_window_width, in_start=input_sequence_start)
    test_x3, test_y3 = sliding_window(test3, sliding_window_width, in_start=input_sequence_start)
    test_x4, test_y4 = sliding_window(test4, sliding_window_width, in_start=input_sequence_start)
    test_x5, test_y5 = sliding_window(test5, sliding_window_width, in_start=input_sequence_start)
    test_x6, test_y6 = sliding_window(test6, sliding_window_width, in_start=input_sequence_start)

    # TODO 去掉了5的输入 train_x5, train_y5,
    # 拼接滑动窗口结果 多维数组拼接 x=> [2166,15,60]  y=>[2166,15,10]
    train_x= np.concatenate((train_x1,train_x2,train_x3,train_x4,train_x6),axis = 0)
    train_y = np.concatenate((train_y1, train_y2, train_y3, train_y4, train_y6), axis=0)
    # TODO train,test数据归一化(注意为常量的bulk density，不进行缩放) MinMaxScaler只能用于一/二维数据
    scaler_trainx = MinMaxScaler(feature_range=(0,1))
    train_X = scaler_trainx.fit_transform(train_x)
    scaler_trainy = MinMaxScaler(feature_range=(0,1))
    train_Y = scaler_trainy.fit_transform(train_y)
    train_X[:,31:40] =train_x[:,31:40]
    train = [train_X,train_Y]

    # TODO 去掉了5的输入 test_x5, test_y5,
    test_x= np.concatenate((test_x1,test_x2,test_x3,test_x4,test_x6),axis = 0)
    test_y = np.concatenate((test_y1, test_y2, test_y3, test_y4, test_y6), axis=0)
    scaler_testx = MinMaxScaler(feature_range=(0,1))
    test_X = scaler_testx.fit_transform(test_x)
    scaler_testy = MinMaxScaler(feature_range=(0,1))
    test_Y = scaler_testy.fit_transform(test_y)
    test_Y[:,31:40] =test_y[:,31:40]
    test=[test_X,test_Y]


    # 如果test
    name = 'LSTM'

    epochs_num = 2
    batch_size_set = 10
    verbose_set = 2  # 显示每个epoch的记录

    main_run(train,test,scaler_trainy,scaler_testy, sliding_window_width, name, input_sequence_start,
             verbose_set, epochs_num, batch_size_set)

    # 预测
    model = load_model('./save_model/LSTM.008-0.0570.h5')
    predicitions = forecast(model,test)
    # TODO　预测结果反归一化
    predicts = scaler_testy.inverse_transform(predicitions)





