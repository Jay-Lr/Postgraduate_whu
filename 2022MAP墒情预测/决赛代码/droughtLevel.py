"""
    根据历史土壤湿度确定分位数，划分干旱等级
"""
import numpy as np
import pandas as pd


# 划分百分位，干旱等级计算
"""
    0-2% Exceptional Drought D4
    3%-5% Extreme Drought D3
    6%-10% Severe Drought D2
    11%-20% Moderate Drought D1
    21%-30% Abnormally Dry D0
    
"""
def daymean(path):
    P1 = pd.read_csv(path)
    P1['时间'] = pd.to_datetime(P1['时间'])
    P1.set_index("时间", inplace=True)
    P1_day = P1.resample('D', label='left').mean()
    # 计算连续三天的均值
    P1_mean = P1_day.rolling(3, center=True).mean()
    return  P1_mean

P1=daymean("./data/checkData1/STSM1.csv")
P2 = daymean("./data/checkData1/STSM2.csv")
P3 = daymean("./data/checkData1/STSM3.csv")
P4 = daymean("./data/checkData1/STSM4.csv")
P5_1 = daymean("./data/checkData1/STSM5_1.csv")
P5_2 = daymean("./data/checkData1/STSM5_2.csv")
P5_3 = daymean("./data/checkData1/STSM5_3.csv")
P5 = pd.concat([P5_1,P5_2,P5_3])
P6_1 = daymean("./data/checkData1/STSM6_1.csv")
# P6_2 = daymean("./data/checkData1/STSM6_2.csv") # 第一层有异常值
P6 = pd.concat([P6_1])





print()


# 根据《农业干旱等级》干旱指数计算
"""
    rsm = a*(wi/fc*100%)/n
    a=1
    
    n=5
"""

def droughtLevel(df,fc,df_soil,soil,point):
    # 作物发育期调节系数
    a = 1
    # 土壤厚度层数（10cm为划分单位）
    n = 5
    # 求前5层的和
    m = 0
    df = df.iloc[0:, 1:]
    for i in range(0,5):
        m = m + df.iloc[:, i] / fc.iloc[i,0]*100
    Rsm = a * m / n
    Rsm = pd.DataFrame(Rsm)
    # 根据前5层土质 和 Rsm判断干旱程度
    if(soil=="砂土"):
        # 砂土
        conditions = [Rsm.iloc[:,0] >= 55,
                    (Rsm.iloc[:,0] < 55) & (Rsm.iloc[:,0] >= 45),
                      (Rsm.iloc[:,0] >= 35) & (Rsm.iloc[:,0] < 45),
                      (Rsm.iloc[:,0] >= 25) & (Rsm.iloc[:,0] < 35),
                      (Rsm.iloc[:,0] < 25)]
        values = ['正常','轻旱', '中旱', '重旱', '特旱']
        Rsm['droughtLevel'] = np.select(conditions, values)
        Rsm.to_csv("./data/submitData/droughtLevel"+str(point)+".csv",encoding='utf-8-sig')
    if(soil=="壤土"):
        # 壤土
        conditions = [Rsm.iloc[:,0] >= 60,
            (Rsm.iloc[:,0] >= 50) & (Rsm.iloc[:,0] < 60),
                      (Rsm.iloc[:,0] >= 40) & (Rsm.iloc[:,0] < 50),
                      (Rsm.iloc[:,0] >= 30) & (Rsm.iloc[:,0] < 40),
                      (Rsm[0] < 30)]
        values = ['正常','轻旱', '中旱', '重旱', '特旱']
        Rsm['droughtLevel'] = np.select(conditions, values)
        Rsm.to_csv("./data/submitData/droughtLevel"+str(point)+".csv",encoding='utf-8-sig')
    return

df1 = pd.read_excel(r'.\data\submitData\predictions_ave.xlsx', sheet_name='预测点1')
df2 = pd.read_excel(r'.\data\submitData\predictions_ave.xlsx', sheet_name='预测点2')
df3 = pd.read_excel(r'.\data\submitData\predictions_ave.xlsx', sheet_name='预测点3')
df4 = pd.read_excel(r'.\data\submitData\predictions_ave.xlsx', sheet_name='预测点4')
df5 = pd.read_excel(r'.\data\submitData\predictions_ave.xlsx', sheet_name='预测点5')
df6 = pd.read_excel(r'.\data\submitData\predictions_ave.xlsx', sheet_name='预测点6')
df1_soil=["砂土","壤土","壤土","壤土","壤土","壤土","壤土","壤土","壤土","砂土"]
df2_soil=["砂土","砂土","砂土","壤土","壤土","壤土","壤土","壤土","壤土","壤土"]
df3_soil=["壤土","壤土","壤土","砂土","壤土","壤土","壤土","壤土","壤土","壤土"]
df4_soil=["砂土","砂土","壤土","壤土","壤土","砂土","砂土","砂土","壤土","壤土"]
df5_soil=["壤土","壤土","壤土","壤土","壤土","壤土","壤土","壤土","壤土","砂土"]
df6_soil=["壤土","壤土","壤土","壤土","壤土","壤土","壤土","砂土","砂土","壤土"]
# 除去第一行和最后一行，去最后10列的最大值
fc1=pd.DataFrame(P1.iloc[1:-1,-10:].max())
fc2=pd.DataFrame(P2.iloc[1:-1,-10:].max())
fc3=pd.DataFrame(P3.iloc[1:-1,-10:].max())
fc4=pd.DataFrame(P4.iloc[1:-1,-10:].max())
fc5=pd.DataFrame(P5.iloc[1:-1,-10:].max())
fc6=pd.DataFrame(P6.iloc[1:-1,-10:].max())


droughtLevel(df1,fc1,df1_soil,soil="壤土",point=1)
droughtLevel(df2,fc2,df2_soil,soil="砂土",point=2)
droughtLevel(df3,fc3,df3_soil,soil="壤土",point=3)
droughtLevel(df4,fc4,df4_soil,soil="壤土",point=4)
droughtLevel(df5,fc5,df5_soil,soil="壤土",point=5)
# droughtLevel(df6,fc6,df6_soil,soil="壤土",point=6) # 注意除去第一层

