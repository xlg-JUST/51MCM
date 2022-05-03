import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
import datetime
from pandas.tseries.offsets import Hour
from Q1Q2 import *
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
import pickle as pkl
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score


def split_data_Q3(data):
    """
    将数据集划分为输入数据与标签数据
    :param data:
    :return:
    """
    t1, t2 = data['Temperature of system I'].to_numpy().reshape(-1), data['Temperature of system II'].to_numpy().reshape(-1)
    m1, m2, m3, m4 = data['Mineral_parameter_1'].to_numpy().reshape(-1), data['Mineral_parameter_2'].to_numpy().reshape(-1), data['Mineral_parameter_3'].to_numpy().reshape(-1), data['Mineral_parameter_4'].to_numpy().reshape(-1)
    A, B, C, D = data['index A'].to_numpy().reshape(-1), data['index B'].to_numpy().reshape(-1), data['index C'].to_numpy().reshape(-1), data['index D'].to_numpy().reshape(-1)
    process3, process4 = data['Process parameter 3'].to_numpy().reshape(-1), data['Process parameter 4'].to_numpy().reshape(-1)
    data = np.array([t1, t2, m1, m2, m3, m4]).T
    index_labels = np.array([A, B, C, D]).T
    process_labels = np.array([process3, process4]).T  # 新增
    return data, index_labels, process_labels


def get_k(index):
    if index == 0:
        return 1
    elif index == 1:
        return 3
    else:
        return 5


def calc_passrate(index_table):
    index_table[0, :] = np.int32(index_table[0, :] < 80.33) * np.int32(index_table[0, :] > 77.78)
    index_table[1, :] = np.int32(index_table[1, :] < 24.15)
    index_table[2, :] = np.int32(index_table[2, :] < 17.15)
    index_table[3, :] = np.int32(index_table[3, :] < 15.62)
    index_table = np.int32(np.sum(index_table, axis=0) / 4)  # 全部满足条件相加才等于4，除以4后等于1

    return np.argwhere(index_table == 1).shape[0]/index_table.shape[0]




if __name__ == '__main__':

    # # 将quality的时间全部提前2小时
    # df = quality_df = pd.read_csv(r'dataset/Att2-quality.csv')
    # df['Time'] = pd.to_datetime(df['Time'])
    # for i in range(df['Time'].shape[0]):
    #     df.loc[i, 'Time'] = df.loc[i, 'Time'] - Hour(2)
    # df.to_csv(r'dataset/Att2-quality.csv', index=False)

    # # 根据时间连接多个表
    # main_df = pd.read_csv(r'dataset/Att2-temperature.csv')
    # quality_df = pd.read_csv(r'dataset/Att2-quality.csv')
    # para_df = pd.read_csv(r'dataset/Att2-parameter.csv')
    # process_df = pd.read_csv(r'dataset/Att2-process.csv')
    #
    # main_df['Time'] = pd.to_datetime(main_df['Time'])
    # quality_df['Time'] = pd.to_datetime(quality_df['Time'])
    # para_df['Time'] = pd.to_datetime(para_df['Time']).dt.date  # 参数根据日期来连接
    # para_df.rename(columns={'Time': 'Date'}, inplace=True)  # 修改下名称，不然与para连接的时候会有两个Time出来，导致process连接报错
    # process_df['Time'] = pd.to_datetime(process_df['Time'])
    # main_df['Date'] = pd.to_datetime(main_df['Time']).dt.date  # 主表新增一个日期方便para连接。事后需要删除
    #
    # main_df = pd.merge(main_df, quality_df, left_on='Time', right_on='Time', how='left', sort=False)
    # main_df = pd.merge(main_df, para_df, left_on='Date', right_on='Date', how='left', sort=False)
    # main_df = pd.merge(main_df, process_df, left_on='Time', right_on='Time', how='left', sort=False)
    #
    # main_df.to_csv(r'dataset/data_resultQ3Q4/data.csv', index=False)
    # # 完事之后，手动删除Date列
    # # 手动删除过程数据1和2，这两列没什么用

    # df = pd.read_csv(r'dataset/data_resultQ3Q4/data.csv')
    # # a = np.int32(df['Temperature of system II'].isnull().to_numpy())
    # # print(sum(a))  # 检测温度2中是否存在空值
    # train_idx = df['Temperature of system I'].notnull()
    # train_data = df[train_idx]
    # train_data = train_data.drop(labels=['Time', 'index A', 'index B', 'index C', 'index D', 'Process parameter 3', 'Process parameter 4'], axis=1).to_numpy()
    # x_data, y_data = train_data[:, 1:], train_data[:, 0]
    # x_data = scale(x_data, axis=1)
    # x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1)
    # model = get_XGBmodel()
    # model.fit(x_train, y_train)
    # y_pred = model.predict(x_test)
    # print(calc_loss(y_test, y_pred))
    # 预测温度1误差太大，无法实现，因此将温度1的空值行删除
    # df = pd.read_csv(r'dataset/data_resultQ3Q4/data.csv')
    # null_idx = df['Temperature of system I'].isnull()
    # null_idx = np.argwhere(np.int32(null_idx) == 1).reshape(-1)
    # df = df.drop(index=null_idx, axis=0)
    # df.to_csv(r'dataset/data_resultQ3Q4/data0.csv', index=False)

    # # 标准化
    # df = pd.read_csv(r'dataset/data_resultQ3Q4/data0.csv')
    # df['Temperature of system I'] = scale(df['Temperature of system I'])
    # df['Temperature of system II'] = scale(df['Temperature of system II'])
    # df['Mineral_parameter_1'] = scale(df['Mineral_parameter_1'])
    # df['Mineral_parameter_2'] = scale(df['Mineral_parameter_2'])
    # df['Mineral_parameter_3'] = scale(df['Mineral_parameter_3'])
    # df['Mineral_parameter_4'] = scale(df['Mineral_parameter_4'])
    # df.to_csv(r'dataset/data_resultQ3Q4/scale_data.csv', index=False)

    # 开始对process数据插值
    # df = pd.read_csv(r'dataset/data_resultQ3Q4/scale_data.csv')
    # idx_header = ['index A', 'index B', 'index C', 'index D']
    # pro_header = ['Process parameter 3', 'Process parameter 4']
    # for i in range(20):  # 先对两个过程数据进行插值
    #     train_index = df['Process parameter 3'].notnull()
    #     train_data = df[train_index]
    #     train_index = np.argwhere(np.int32(train_index) == 1).reshape(-1)  # 转成了numpy 方便找前后的target数据
    #     tar_idx = get_tar_idx(train_index, k=get_k(i))
    #     tar_data = df.loc[tar_idx, :]
    #     train_data, _, train_pro_labels = split_data_Q3(train_data)
    #     tar_data, _, _ = split_data_Q3(tar_data)
    #     for j in range(2):
    #         model = get_XGBmodel()
    #         x_train = train_data
    #         y_train = train_pro_labels[:, j]
    #         model.fit(x_train, y_train)
    #         predict = model.predict(tar_data).reshape(-1)
    #         assert tar_idx.shape[0] == predict.shape[0]
    #         # for k in range(tar_idx.shape[0]):
    #         #     df.loc[tar_idx[k], pro_header[j]] = predict[k]
    #         df.loc[tar_idx, pro_header[j]] = predict  # 直接写，不用做循环
    #
    # df.to_csv(r'dataset/data_resultQ3Q4/tmp.csv')

    # pro_header = ['Process parameter 3', 'Process parameter 4']
    # df = pd.read_csv(r'dataset/data_resultQ3Q4/tmp.csv')
    # train_index = df['Process parameter 3'].notnull()
    # train_data = df[train_index]
    # tar_idx = df['Process parameter 3'].isnull()
    # tar_data = df[tar_idx]
    # tar_idx = np.argwhere(np.int32(tar_idx) == 1).reshape(-1)  # 这里==1 实际找的是空值，因为上面使用的是isnull() 很烦
    # train_data, _, train_pro_labels = split_data_Q3(train_data)
    # tar_data, _, _ = split_data_Q3(tar_data)
    # for j in range(2):
    #     model = get_Linearmodel()
    #     x_train = train_data  # 为了好看点。。
    #     print(np.argwhere(np.isnan(x_train)))
    #     y_train = train_pro_labels[:, j].reshape(-1)
    #     model.fit(x_train, y_train)  # shut up!
    #     predict = model.predict(tar_data).reshape(-1)
    #     assert tar_idx.shape[0] == predict.shape[0]
    #     df.loc[tar_idx, pro_header[j]] = predict  # 直接写，不用做循环
    #
    # df.to_csv(r'dataset/data_resultQ3Q4/ProcessDone.csv', index=False)

    # 开始对index数据插值
    # df = pd.read_csv(r'dataset/data_resultQ3Q4/data0.csv')
    # df['Temperature of system I'] = scale(df['Temperature of system I'])
    # df['Temperature of system II'] = scale(df['Temperature of system II'])
    # df['Mineral_parameter_1'] = scale(df['Mineral_parameter_1'])
    # df['Mineral_parameter_2'] = scale(df['Mineral_parameter_2'])
    # df['Mineral_parameter_3'] = scale(df['Mineral_parameter_3'])
    # df['Mineral_parameter_4'] = scale(df['Mineral_parameter_4'])
    # df['Process parameter 3'] = scale(df['Process parameter 3'])
    # df['Process parameter 4'] = scale(df['Process parameter 4'])
    # df.to_csv(r'dataset/data_resultQ3Q4/scale_data0.csv', index=False)

    # df = pd.read_csv(r'dataset/data_resultQ3Q4/scale_data0.csv')
    # idx_header = ['index A', 'index B', 'index C', 'index D']
    # pro_header = ['Process parameter 3', 'Process parameter 4']
    # for i in range(20):  # 再对4个指标进行插值
    #     train_index = df['index A'].notnull()
    #     train_data = df[train_index]
    #     train_index = np.argwhere(np.int32(train_index) == 1).reshape(-1)  # 转成了numpy 方便找前后的target数据
    #     tar_idx = get_tar_idx(train_index, k=get_k(i))
    #     tar_data = df.loc[tar_idx, :]
    #     train_data, train_idx_labels, train_pro_labels = split_data_Q3(train_data)
    #     train_data = np.hstack([train_data, train_pro_labels])  # 过程数据现已加入豪华午餐
    #     tar_data, _, tar_pro_labels = split_data_Q3(tar_data)
    #     tar_data = np.hstack([tar_data, tar_pro_labels])
    #     for j in range(4):
    #         model = get_XGBmodel()
    #         x_train = train_data
    #         y_train = train_idx_labels[:, j]
    #         model.fit(x_train, y_train)
    #         predict = model.predict(tar_data).reshape(-1)
    #         assert tar_idx.shape[0] == predict.shape[0]
    #         # for k in range(tar_idx.shape[0]):
    #         #     df.loc[tar_idx[k], pro_header[j]] = predict[k]
    #         df.loc[tar_idx, idx_header[j]] = predict  # 直接写，不用做循环
    #
    # df.to_csv(r'dataset/data_resultQ3Q4/tmp.csv')

    # idx_header = ['index A', 'index B', 'index C', 'index D']
    # pro_header = ['Process parameter 3', 'Process parameter 4']
    # df = pd.read_csv(r'dataset/data_resultQ3Q4/tmp.csv')
    # train_index = df['index A'].notnull()
    # train_data = df[train_index]
    # tar_idx = df['index A'].isnull()
    # tar_data = df[tar_idx]
    # tar_idx = np.argwhere(np.int32(tar_idx) == 1).reshape(-1)  # 这里==1 实际找的是空值，因为上面使用的是isnull() 很烦
    # train_data, train_idx_labels, train_pro_labels = split_data_Q3(train_data)
    # train_data = np.hstack([train_data, train_pro_labels])  # 过程数据现已加入豪华午餐
    # tar_data, _, tar_pro_labels = split_data_Q3(tar_data)
    # tar_data = np.hstack([tar_data, tar_pro_labels])
    # for j in range(4):
    #     model = get_XGBmodel()
    #     x_train = train_data
    #     y_train = train_idx_labels[:, j]
    #     model.fit(x_train, y_train)
    #     predict = model.predict(tar_data).reshape(-1)
    #     assert tar_idx.shape[0] == predict.shape[0]
    #     # for k in range(tar_idx.shape[0]):
    #     #     df.loc[tar_idx[k], pro_header[j]] = predict[k]
    #     df.loc[tar_idx, idx_header[j]] = predict  # 直接写，不用做循环
    #
    # df.to_csv(r'dataset/data_resultQ3Q4/Done.csv', index=False)
    # 手动改正了Done的数据，把标准化的数据全部换成了非标准化数据

    # 生成Q3基本数据
    # time = pd.date_range('2022/4/8', '2022/4/10', freq='min')
    # df = pd.DataFrame(data=time, index=None, columns=['Time'])
    # df.to_csv(r'dataset/data_resultQ3Q4/Q3.csv', index=False)

    # df = pd.read_csv(r'dataset/data_resultQ3Q4/Q3.csv')
    # df['Time'] = pd.to_datetime(df['Time'])
    # df['Date'] = df['Time'].dt.date
    # df['Date'] = pd.to_datetime(df['Date'])
    # df_temp = pd.read_csv(r'dataset/data_resultQ3Q4/Q3temp.csv')
    # df_temp['Date'] = pd.to_datetime(df_temp['Date'])
    # result = pd.merge(df, df_temp, left_on='Date', right_on='Date', how='left', sort=False)
    # result.to_csv(r'dataset/data_resultQ3Q4/Q3_0.csv', index=False)

    # df = pd.read_csv('dataset/data_resultQ3Q4/Q3_0.csv')
    # df_process = pd.read_csv(r'dataset/Att2-process.csv')
    # result = pd.merge(df, df_process, left_on='Time', right_on='Time', how='left', sort=False)
    # result.to_csv('dataset/data_resultQ3Q4/Q3_1.csv', index=False)

    # df = pd.read_csv('dataset/data_resultQ3Q4/Q3_1.csv')
    # df['Temperature of system I'] = scale(df['Temperature of system I'])
    # df['Temperature of system II'] = scale(df['Temperature of system II'])
    # df['Mineral_parameter_1'] = scale(df['Mineral_parameter_1'])
    # df['Mineral_parameter_2'] = scale(df['Mineral_parameter_2'])
    # df['Mineral_parameter_3'] = scale(df['Mineral_parameter_3'])
    # df['Mineral_parameter_4'] = scale(df['Mineral_parameter_4'])
    # df.to_csv(r'dataset/data_resultQ3Q4/Q3_1_scale.csv', index=False)

    # # 由Done文件构建出两个基础模型，后根据Q3_1的数据分别进行微调与插值 model0->para3  model1->para4
    # main_df = pd.read_csv(r'dataset/data_resultQ3Q4/Done.csv')
    # scaler = StandardScaler()
    # train_data, _, train_pro_labels = split_data_Q3(main_df)
    # train_data = scaler.fit_transform(train_data)
    # model1 = get_XGBmodel()
    # model1.fit(train_data, train_pro_labels[:, 0])
    # with open(r'dataset/data_resultQ3Q4/model0.pkl', 'wb') as f:
    #     pkl.dump(model1, f)
    # f.close()
    # model2 = get_XGBmodel()
    # model2.fit(train_data, train_pro_labels[:, 1])
    # with open(r'dataset/data_resultQ3Q4/model1.pkl', 'wb') as f:
    #     pkl.dump(model2, f)
    # f.close()


    # main_df = pd.read_csv(r'dataset/data_resultQ3Q4/Done.csv')
    # scaler = StandardScaler()
    # train_data, _, _ = split_data_Q3(main_df)
    # scaler.fit(train_data)
    # df = pd.read_csv(r'dataset/data_resultQ3Q4/Q3_1.csv')
    # tar_data, _, _ = split_data_Q3(df)
    # tar_data = scaler.transform(tar_data)
    #
    # idx_header = ['index A', 'index B', 'index C', 'index D']
    # pro_header = ['Process parameter 3', 'Process parameter 4']
    # for i in range(10):  # 先对两个过程数据进行插值
    #     train_index = df['Process parameter 3'].notnull()
    #     train_data = df[train_index]
    #     train_index = np.argwhere(np.int32(train_index) == 1).reshape(-1)  # 转成了numpy 方便找前后的target数据
    #     tar_idx = get_tar_idx(train_index, k=get_k(i))
    #     tar_data = df.loc[tar_idx, :]
    #     train_data, _, train_pro_labels = split_data_Q3(train_data)
    #     tar_data, _, _ = split_data_Q3(tar_data)
    #     for j in range(2):
    #         model = pkl.load(open('dataset/data_resultQ3Q4/model{}.pkl'.format(j), 'rb'))
    #         x_train = train_data
    #         y_train = train_pro_labels[:, j]
    #         model.fit(x_train, y_train)
    #         predict = model.predict(tar_data).reshape(-1)
    #         assert tar_idx.shape[0] == predict.shape[0]
    #         # for k in range(tar_idx.shape[0]):
    #         #     df.loc[tar_idx[k], pro_header[j]] = predict[k]
    #         df.loc[tar_idx, pro_header[j]] = predict  # 直接写，不用做循环
    #
    # pro_header = ['Process parameter 3', 'Process parameter 4']
    # train_index = df['Process parameter 3'].notnull()
    # train_data = df[train_index]
    # tar_idx = df['Process parameter 3'].isnull()
    # tar_data = df[tar_idx]
    # tar_idx = np.argwhere(np.int32(tar_idx) == 1).reshape(-1)  # 这里==1 实际找的是空值，因为上面使用的是isnull() 很烦
    # train_data, _, train_pro_labels = split_data_Q3(train_data)
    # tar_data, _, _ = split_data_Q3(tar_data)
    # for j in range(2):
    #     model = pkl.load(open('dataset/data_resultQ3Q4/model{}.pkl'.format(j), 'rb'))
    #     x_train = train_data  # 为了好看点。。
    #     print(np.argwhere(np.isnan(x_train)))
    #     y_train = train_pro_labels[:, j].reshape(-1)
    #     model.fit(x_train, y_train)  # shut up!
    #     predict = model.predict(tar_data).reshape(-1)
    #     assert tar_idx.shape[0] == predict.shape[0]
    #     df.loc[tar_idx, pro_header[j]] = predict  # 直接写，不用做循环
    #
    # df.to_csv(r'dataset/data_resultQ3Q4/Q3_ProcessDone.csv', index=False)


    # df_train = pd.read_csv('dataset/data_resultQ3Q4/Done.csv')
    # df_test = pd.read_csv('dataset/data_resultQ3Q4/Q3_ProcessDone.csv')
    # pro_header = ['Process parameter 3', 'Process parameter 4']
    # idx_header = ['index A', 'index B', 'index C', 'index D']
    # train_data, train_idx_labels, train_pro_labels = split_data_Q3(df_train)
    # train_data = np.hstack([train_data, train_pro_labels])
    # tar_data, _, tar_pro_labels = split_data_Q3(df_test)
    # tar_data = np.hstack([tar_data, tar_pro_labels])
    # scaler = StandardScaler()
    # train_data = scaler.fit_transform(train_data)
    # tar_data = scaler.transform(tar_data)
    # for i in range(4):
    #     model = get_XGBmodel()
    #     model.fit(train_data, train_idx_labels[:, i])
    #     predict = model.predict(tar_data)
    #     df_test.loc[:, idx_header[i]] = predict  # 直接写，不用做循环
    # df_test.to_csv('dataset/data_resultQ3Q4/Q3_Done.csv', index=False)
    # 这方法做出来合格率为0，插值效果也不好，现在试试不进行插值，过程数据用一次函数替代

    # df = pd.read_csv('dataset/data_resultQ3Q4/Q3_1.csv')
    # pp3 = df['Process parameter 3']
    # pp4 = df['Process parameter 4']
    # pp3notnull = np.argwhere(np.int32(pp3.notnull()) == 1).reshape(-1)
    # list1, list2 = pp3notnull[:-1], pp3notnull[1:]
    # for elm1, elm2 in zip(list1, list2):
    #     step = elm2-elm1
    #     pp3basic = pp3[elm1]
    #     pp4basic = pp4[elm1]
    #     pp3delta = (pp3[elm2]-pp3[elm1]) / step
    #     pp4delta = (pp4[elm2] - pp4[elm1]) / step
    #     for i in range(step):
    #         pp3basic += pp3delta
    #         pp3[elm1+1+i] = pp3basic
    #         pp4basic += pp4delta
    #         pp4[elm1+1+i] = pp4basic
    # df.to_csv('dataset/data_resultQ3Q4/Q3_2.csv', index=False)

    # idx_header = ['index A', 'index B', 'index C', 'index D']
    # train_df = pd.read_csv(r'dataset/data_resultQ3Q4/Done.csv')
    # test_df = pd.read_csv('dataset/data_resultQ3Q4/Q3_2.csv')
    #
    # scaler = StandardScaler()
    # train_data, train_idx_labels, train_pro_labels = split_data_Q3(train_df)
    # train_data = np.hstack([train_data, train_pro_labels])
    # train_data = scaler.fit_transform(train_data)
    #
    # tar_data, _, tar_pro_labels = split_data_Q3(test_df)
    # tar_data = np.hstack([tar_data, tar_pro_labels])
    #
    # tar_data = scaler.transform(tar_data)
    # for i in range(4):
    #     x_train = train_data
    #     y_train = train_idx_labels[:, i]
    #     x_test = tar_data
    #     model = get_XGBmodel()
    #     model.fit(x_train, y_train)
    #     predict = model.predict(x_test).reshape(-1)
    #     test_df.loc[:, idx_header[i]] = predict
    # test_df.to_csv('dataset/data_resultQ3Q4/Q3_Done.csv', index=False)

    # 这部分是计算合格率的
    # df = pd.read_csv(r'dataset/data_resultQ3Q4/Q3_Done.csv')
    # df['Time'] = pd.to_datetime(df['Time'])
    # second_day = df['Time'] >= '2022/4/9'
    # first_day = ~second_day
    # first_day = df[first_day]
    # second_day = df[second_day]
    # first_all = first_day.shape[0]
    # second_all = second_day.shape[0]
    # first_index = first_day.loc[:, ('index A', 'index B', 'index C', 'index D')].to_numpy().T
    # second_index = second_day.loc[:, ('index A', 'index B', 'index C', 'index D')].to_numpy().T
    # first_index[0, :] = np.int32(first_index[0, :] < 80.33) * np.int32(first_index[0, :] > 77.78)
    # first_index[1, :] = np.int32(first_index[1, :] < 24.15)
    # first_index[2, :] = np.int32(first_index[2, :] < 17.15)
    # first_index[3, :] = np.int32(first_index[3, :] < 15.62)
    # first_index = np.int32(np.sum(first_index, axis=0) / 4)  # 全部满足条件相加才等于4，除以4后等于1
    # print(np.argwhere(first_index == 1).shape[0]/first_all)
    #
    # second_index[0, :] = np.int32(second_index[0, :] < 80.33) * np.int32(second_index[0, :] > 77.78)
    # second_index[1, :] = np.int32(second_index[1, :] < 24.15)
    # second_index[2, :] = np.int32(second_index[2, :] < 17.15)
    # second_index[3, :] = np.int32(second_index[3, :] < 15.62)
    # second_index = np.int32(np.sum(second_index, axis=0) / 4)  # 全部满足条件相加才等于4，除以4后等于1
    # print(np.argwhere(second_index == 1).shape[0]/second_all)

    # 上面方法合格率预测太低，明显有问题，重头做，全部假设过程数据是线性的
    # df = pd.read_csv(r'dataset/data_resultQ3Q4/data.csv')
    # null_idx = df['Temperature of system I'].isnull()
    # null_idx = np.argwhere(np.int32(null_idx) == 1).reshape(-1)
    # df = df.drop(index=null_idx, axis=0)
    # df.to_csv(r'dataset/data_resultQ3Q4/data0.csv', index=False)

    # 开始process插值
    # df = pd.read_csv(r'dataset/data_resultQ3Q4/data0.csv')
    # pp3 = df['Process parameter 3']
    # pp4 = df['Process parameter 4']
    # pp3notnull = np.argwhere(np.int32(pp3.notnull()) == 1).reshape(-1)
    # list1, list2 = pp3notnull[:-1], pp3notnull[1:]
    # for elm1, elm2 in zip(list1, list2):
    #     step = elm2-elm1
    #     pp3basic = pp3[elm1]
    #     pp4basic = pp4[elm1]
    #     pp3delta = (pp3[elm2]-pp3[elm1]) / step
    #     pp4delta = (pp4[elm2] - pp4[elm1]) / step
    #     for i in range(step):
    #         pp3basic += pp3delta
    #         pp3[elm1+1+i] = pp3basic
    #         pp4basic += pp4delta
    #         pp4[elm1+1+i] = pp4basic
    # df.to_csv('dataset/data_resultQ3Q4/data0_processDone.csv', index=False)

    # 开始index预测插值
    # idx_header = ['index A', 'index B', 'index C', 'index D']
    # pro_header = ['Process parameter 3', 'Process parameter 4']
    # df = pd.read_csv(r'dataset/data_resultQ3Q4/data0_processDone.csv')
    # df['Temperature of system I'] = scale(df['Temperature of system I'])
    # df['Temperature of system II'] = scale(df['Temperature of system II'])
    # # 标准化
    # df['Mineral_parameter_1'] = scale(df['Mineral_parameter_1'])
    # df['Mineral_parameter_2'] = scale(df['Mineral_parameter_2'])
    # df['Mineral_parameter_3'] = scale(df['Mineral_parameter_3'])
    # df['Mineral_parameter_4'] = scale(df['Mineral_parameter_4'])
    # df['Process parameter 3'] = scale(df['Process parameter 3'])
    # df['Process parameter 4'] = scale(df['Process parameter 4'])
    #
    # for i in range(20):  # 对4个指标进行插值
    #     train_index = df['index A'].notnull()
    #     train_data = df[train_index]
    #     train_index = np.argwhere(np.int32(train_index) == 1).reshape(-1)  # 转成了numpy 方便找前后的target数据
    #     tar_idx = get_tar_idx(train_index, k=get_k(i))
    #     tar_data = df.loc[tar_idx, :]
    #     train_data, train_idx_labels, train_pro_labels = split_data_Q3(train_data)
    #     train_data = np.hstack([train_data, train_pro_labels])  # 过程数据现已加入豪华午餐
    #     tar_data, _, tar_pro_labels = split_data_Q3(tar_data)
    #     tar_data = np.hstack([tar_data, tar_pro_labels])
    #     for j in range(4):
    #         model = get_XGBmodel()
    #         x_train = train_data
    #         y_train = train_idx_labels[:, j]
    #         model.fit(x_train, y_train)
    #         predict = model.predict(tar_data).reshape(-1)
    #         assert tar_idx.shape[0] == predict.shape[0]
    #         # for k in range(tar_idx.shape[0]):
    #         #     df.loc[tar_idx[k], pro_header[j]] = predict[k]
    #         df.loc[tar_idx, idx_header[j]] = predict  # 直接写，不用做循环
    #
    # idx_header = ['index A', 'index B', 'index C', 'index D']
    # pro_header = ['Process parameter 3', 'Process parameter 4']
    # train_index = df['index A'].notnull()
    # train_data = df[train_index]
    # tar_idx = df['index A'].isnull()
    # tar_data = df[tar_idx]
    # tar_idx = np.argwhere(np.int32(tar_idx) == 1).reshape(-1)  # 这里==1 实际找的是空值，因为上面使用的是isnull() 很烦
    # train_data, train_idx_labels, train_pro_labels = split_data_Q3(train_data)
    # train_data = np.hstack([train_data, train_pro_labels])  # 过程数据现已加入豪华午餐
    # tar_data, _, tar_pro_labels = split_data_Q3(tar_data)
    # tar_data = np.hstack([tar_data, tar_pro_labels])
    # for j in range(4):
    #     model = get_XGBmodel()
    #     x_train = train_data
    #     y_train = train_idx_labels[:, j]
    #     model.fit(x_train, y_train)
    #     predict = model.predict(tar_data).reshape(-1)
    #     assert tar_idx.shape[0] == predict.shape[0]
    #     # for k in range(tar_idx.shape[0]):
    #     #     df.loc[tar_idx[k], pro_header[j]] = predict[k]
    #     df.loc[tar_idx, idx_header[j]] = predict  # 直接写，不用做循环
    #
    # df.to_csv(r'dataset/data_resultQ3Q4/Done.csv', index=False)

    # idx_header = ['index A', 'index B', 'index C', 'index D']
    # train_df = pd.read_csv(r'dataset/data_resultQ3Q4/Done.csv')
    # test_df = pd.read_csv('dataset/data_resultQ3Q4/Q3_2.csv')
    #
    # scaler = StandardScaler()
    # train_data, train_idx_labels, train_pro_labels = split_data_Q3(train_df)
    # train_data = np.hstack([train_data, train_pro_labels])
    # train_data = scaler.fit_transform(train_data)
    #
    # tar_data, _, tar_pro_labels = split_data_Q3(test_df)
    # tar_data = np.hstack([tar_data, tar_pro_labels])
    #
    # tar_data = scaler.transform(tar_data)
    # for i in range(4):
    #     x_train = train_data
    #     y_train = train_idx_labels[:, i]
    #     x_test = tar_data
    #     model = get_XGBmodel()
    #     model.fit(x_train, y_train)
    #     predict = model.predict(x_test).reshape(-1)
    #     test_df.loc[:, idx_header[i]] = predict
    # test_df.to_csv('dataset/data_resultQ3Q4/Q3_Done.csv', index=False)

    # df = pd.read_csv(r'dataset/data_resultQ3Q4/Q3_Done.csv')
    # df['Time'] = pd.to_datetime(df['Time'])
    # second_day = df['Time'] >= '2022/4/9'
    # first_day = ~second_day
    # first_day = df[first_day]
    # second_day = df[second_day]
    # first_all = first_day.shape[0]
    # second_all = second_day.shape[0]
    # first_index = first_day.loc[:, ('index A', 'index B', 'index C', 'index D')].to_numpy().T
    # second_index = second_day.loc[:, ('index A', 'index B', 'index C', 'index D')].to_numpy().T
    # first_index[0, :] = np.int32(first_index[0, :] < 80.33) * np.int32(first_index[0, :] > 77.78)
    # first_index[1, :] = np.int32(first_index[1, :] < 24.15)
    # first_index[2, :] = np.int32(first_index[2, :] < 17.15)
    # first_index[3, :] = np.int32(first_index[3, :] < 15.62)
    # first_index = np.int32(np.sum(first_index, axis=0) / 4)  # 全部满足条件相加才等于4，除以4后等于1
    # print(np.argwhere(first_index == 1).shape[0]/first_all)
    #
    # second_index[0, :] = np.int32(second_index[0, :] < 80.33) * np.int32(second_index[0, :] > 77.78)
    # second_index[1, :] = np.int32(second_index[1, :] < 24.15)
    # second_index[2, :] = np.int32(second_index[2, :] < 17.15)
    # second_index[3, :] = np.int32(second_index[3, :] < 15.62)
    # second_index = np.int32(np.sum(second_index, axis=0) / 4)  # 全部满足条件相加才等于4，除以4后等于1
    # print(np.argwhere(second_index == 1).shape[0]/second_all)

    # df = pd.read_csv(r'dataset/Att2-quality.csv').loc[:, ('index A', 'index B', 'index C', 'index D')].to_numpy().T
    # df[0, :] = np.int32(df[0, :] < 80.33) * np.int32(df[0, :] > 77.78)
    # df[1, :] = np.int32(df[1, :] < 24.15)
    # df[2, :] = np.int32(df[2, :] < 17.15)
    # df[3, :] = np.int32(df[3, :] < 15.62)
    # df = np.int32(np.sum(df, axis=0) / 4)  # 全部满足条件相加才等于4，除以4后等于1
    # y_true = np.int32(df == 1)  # 产品是否合格 1合格 0不合格
    # # print(np.argwhere(df == 1).shape[0]/df.shape[0])

    # # 使用插补的数据对原有真实数据进行预测，检测其合格率
    # df = pd.read_csv(r'dataset/data_resultQ3Q4/data0.csv')
    # tar_idx = df['index A'].notnull()
    # train_idx = ~tar_idx
    # df = pd.read_csv(r'dataset/data_resultQ3Q4/Done.csv')
    # train_data = df[train_idx]
    # tar_data = df[tar_idx]
    # train_data, train_idx_labels, train_pro_labels = split_data_Q3(train_data)
    # tar_data, y_true, tar_pro_labels = split_data_Q3(tar_data)
    # train_data = np.hstack([train_data, train_pro_labels])
    # tar_data = np.hstack([tar_data, tar_pro_labels])
    # scaler = StandardScaler()
    # train_data = scaler.fit_transform(train_data)
    # tar_data = scaler.transform(tar_data)
    # predict_labels = []
    # for i in range(4):
    #     model = get_XGBmodel()
    #     model.fit(train_data, train_idx_labels[:, i])
    #     predict = model.predict(tar_data).reshape(-1)
    #     predict_labels.append(predict)
    # predict_labels = np.array(predict_labels)
    # predict_labels[0, :] = np.int32(predict_labels[0, :] < 80.33) * np.int32(predict_labels[0, :] > 77.78)
    # predict_labels[1, :] = np.int32(predict_labels[1, :] < 24.15)
    # predict_labels[2, :] = np.int32(predict_labels[2, :] < 17.15)
    # predict_labels[3, :] = np.int32(predict_labels[3, :] < 15.62)
    # predict_labels = np.int32(np.sum(predict_labels, axis=0) / 4)  # 全部满足条件相加才等于4，除以4后等于1
    # y_pred = np.int32(predict_labels == 1)
    # y_true = y_true.T
    # y_true[0, :] = np.int32(y_true[0, :] < 80.33) * np.int32(y_true[0, :] > 77.78)
    # y_true[1, :] = np.int32(y_true[1, :] < 24.15)
    # y_true[2, :] = np.int32(y_true[2, :] < 17.15)
    # y_true[3, :] = np.int32(y_true[3, :] < 15.62)
    # y_true = np.int32(np.sum(y_true, axis=0) / 4)  # 全部满足条件相加才等于4，除以4后等于1
    # y_true = np.int32(y_true == 1)
    # print('Acc:{}'.format(accuracy_score(y_true, y_pred)))  # 86.65
    # print('Precision:{}'.format(precision_score(y_true, y_pred)))  # 76.60
    # print('Recall:{}'.format(recall_score(y_true, y_pred)))  # 76.76
    # print('F1-Score:{}'.format(f1_score(y_true, y_pred)))  # 76.68



    #Q4
    # time = pd.date_range(start='2022/04/10', end='2022/04/12', freq='min')
    # main_df = pd.DataFrame(time, columns=['Time'], index=None)
    # main_df['Time'] = pd.to_datetime(main_df['Time'])
    # main_df['Date'] = main_df['Time'].dt.date
    # para_df = pd.read_csv(r'dataset/Att2-parameter.csv')
    # process_df = pd.read_csv(r'dataset/Att2-process.csv')
    # para_df['Time'] = pd.to_datetime(para_df['Time'])
    # para_df['Date'] = para_df['Time'].dt.date
    # main_df = pd.merge(main_df, para_df, left_on='Date', right_on='Date', how='left', sort=False)
    # main_df.to_csv('dataset/data_resultQ3Q4/Q4.csv')
    # main_df = pd.read_csv('dataset/data_resultQ3Q4/Q4.csv')
    # main_df = pd.merge(main_df, process_df, left_on='Time', right_on='Time', how='left', sort=False)
    # main_df.to_csv('dataset/data_resultQ3Q4/Q40.csv')
    # 一个一个连

    # df = pd.read_csv('dataset/data_resultQ3Q4/Q4.csv')
    # pp3 = df['Process parameter 3']
    # pp4 = df['Process parameter 4']
    # pp3notnull = np.argwhere(np.int32(pp3.notnull()) == 1).reshape(-1)
    # list1, list2 = pp3notnull[:-1], pp3notnull[1:]
    # for elm1, elm2 in zip(list1, list2):
    #     step = elm2-elm1
    #     pp3basic = pp3[elm1]
    #     pp4basic = pp4[elm1]
    #     pp3delta = (pp3[elm2]-pp3[elm1]) / step
    #     pp4delta = (pp4[elm2] - pp4[elm1]) / step
    #     for i in range(step):
    #         pp3basic += pp3delta
    #         pp3[elm1+1+i] = pp3basic
    #         pp4basic += pp4delta
    #         pp4[elm1+1+i] = pp4basic
    # df.to_csv('dataset/data_resultQ3Q4/Q4_processDone.csv', index=False)

    # # 训练一组分类器待使用，还有标准化装置
    # train_df = pd.read_csv('dataset/data_resultQ3Q4/Done.csv')
    # train_data, train_idx_labels, train_pro_labels = split_data_Q3(train_df)
    # train_data = np.hstack([train_data, train_pro_labels])
    # scaler = StandardScaler()
    # train_data = scaler.fit_transform(train_data)
    # with open(r'dataset/data_resultQ3Q4/Q4scaler.pkl', 'wb') as f:
    #     pkl.dump(scaler, f)
    #     f.close()
    #
    # modelA = get_XGBmodel()
    # modelA.fit(train_data, train_idx_labels[:, 0])
    # with open(r'dataset/data_resultQ3Q4/Q4modelA.pkl', 'wb') as f:
    #     pkl.dump(modelA, f)
    #     f.close()
    #
    # modelB = get_XGBmodel()
    # modelB.fit(train_data, train_idx_labels[:, 1])
    # with open(r'dataset/data_resultQ3Q4/Q4modelB.pkl', 'wb') as f:
    #     pkl.dump(modelB, f)
    #     f.close()
    #
    # modelC = get_XGBmodel()
    # modelC.fit(train_data, train_idx_labels[:, 2])
    # with open(r'dataset/data_resultQ3Q4/Q4modelC.pkl', 'wb') as f:
    #     pkl.dump(modelC, f)
    #     f.close()
    #
    # modelD = get_XGBmodel()
    # modelD.fit(train_data, train_idx_labels[:, 3])
    # with open(r'dataset/data_resultQ3Q4/Q4modelD.pkl', 'wb') as f:
    #     pkl.dump(modelD, f)
    #     f.close()
    modelA = pkl.load(open(r'dataset/data_resultQ3Q4/Q4modelA.pkl', 'rb'))
    modelB = pkl.load(open(r'dataset/data_resultQ3Q4/Q4modelB.pkl', 'rb'))
    modelC = pkl.load(open(r'dataset/data_resultQ3Q4/Q4modelC.pkl', 'rb'))
    modelD = pkl.load(open(r'dataset/data_resultQ3Q4/Q4modelD.pkl', 'rb'))
    scaler = pkl.load(open(r'dataset/data_resultQ3Q4/Q4scaler.pkl', 'rb'))

    models = [modelA, modelB, modelC, modelD]
    header = ['index A', 'index B', 'index C', 'index D']
    df = pd.read_csv(r'dataset/data_resultQ3Q4/Q4_processDone.csv')
    df['Time'] = pd.to_datetime(df['Time'])
    second_day = df['Time'] >= '2022/4/11'
    first_day = ~second_day
    df1 = df[first_day]
    df2 = df[second_day]

    # df1 第一天  df2 第二天
    data = df2.loc[:, ('Mineral_parameter_1', 'Mineral_parameter_2', 'Mineral_parameter_3', 'Mineral_parameter_4', 'Process parameter 3', 'Process parameter 4')]
    init_t1, init_t2 = np.arange(1290, 1310, 1), np.arange(540, 550, 1)
    max_passrate = 0.
    max_t1, max_t2 = 0., 0.
    for t1 in init_t1:
        for t2 in init_t2:
            _t1 = np.array([t1]*data.shape[0]).reshape((-1, 1))
            _t2 = np.array([t2]*data.shape[0]).reshape((-1, 1))
            cur_data = np.hstack([_t1, _t2, data])
            cur_data = scaler.transform(cur_data)
            pred_list = []
            for i in range(4):
                pred_list.append(models[i].predict(cur_data).reshape(-1))
            pred_list = np.vstack(pred_list)
            passrate = calc_passrate(pred_list)
            if max_passrate < passrate:
                max_passrate = passrate
                max_t1, max_t2 = t1, t2
    print(max_passrate)
    print(max_t1)
    print(max_t2)

    # 敏感性分析，基准值：t1：972， t2:818, m1:56, m2:106, m3:47, m4:20  p3:270  p4:150

    # # t1 分析
    # t1 = np.arange(300, 1501, 50)
    # t2 = np.array([818]*t1.shape[0])
    # m1 = np.array([56]*t1.shape[0])
    # m2 = np.array([106]*t1.shape[0])
    # m3 = np.array([47]*t1.shape[0])
    # m4 = np.array([20]*t1.shape[0])
    # p3 = np.array([270]*t1.shape[0])
    # p4 = np.array([150]*t1.shape[0])
    # data = np.vstack([t1, t2, m1, m2, m3, m4, p3, p4]).T
    # data = scaler.transform(data)
    # pred_list = []
    # for i in range(4):
    #     pred_list.append(models[i].predict(data))
    # pred_list = np.vstack(pred_list).T
    # df = pd.DataFrame(pred_list, columns=header, index=t1)
    # df.to_csv(r'dataset/data_resultQ3Q4/sense_t1.csv')

    # t2 分析
    # t2 = np.arange(270, 1201, 40)
    # t1 = np.array([972]*t2.shape[0])
    # m1 = np.array([56]*t2.shape[0])
    # m2 = np.array([106]*t2.shape[0])
    # m3 = np.array([47]*t2.shape[0])
    # m4 = np.array([20]*t2.shape[0])
    # p3 = np.array([270]*t2.shape[0])
    # p4 = np.array([150]*t2.shape[0])
    # data = np.vstack([t1, t2, m1, m2, m3, m4, p3, p4]).T
    # data = scaler.transform(data)
    # pred_list = []
    # for i in range(4):
    #     pred_list.append(models[i].predict(data))
    # pred_list = np.vstack(pred_list).T
    # df = pd.DataFrame(pred_list, columns=header, index=t2)
    # df.to_csv(r'dataset/data_resultQ3Q4/sense_t2.csv')

    # # m1 分析
    # m1 = np.arange(51, 63, 0.5)
    # t1 = np.array([972] * m1.shape[0])
    # t2 = np.array([818]*m1.shape[0])
    # m2 = np.array([106]*m1.shape[0])
    # m3 = np.array([47]*m1.shape[0])
    # m4 = np.array([20]*m1.shape[0])
    # p3 = np.array([270]*m1.shape[0])
    # p4 = np.array([150]*m1.shape[0])
    # data = np.vstack([t1, t2, m1, m2, m3, m4, p3, p4]).T
    # data = scaler.transform(data)
    # pred_list = []
    # for i in range(4):
    #     pred_list.append(models[i].predict(data))
    # pred_list = np.vstack(pred_list).T
    # df = pd.DataFrame(pred_list, columns=header, index=m1)
    # df.to_csv(r'dataset/data_resultQ3Q4/sense_m1.csv')

    # # m2 分析
    # m2 = np.arange(80, 133, 2)
    # t1 = np.array([972]*m2.shape[0])
    # t2 = np.array([818]*m2.shape[0])
    # m1 = np.array([56]*m2.shape[0])
    # m3 = np.array([47]*m2.shape[0])
    # m4 = np.array([20]*m2.shape[0])
    # p3 = np.array([270]*m2.shape[0])
    # p4 = np.array([150]*m2.shape[0])
    # data = np.vstack([t1, t2, m1, m2, m3, m4, p3, p4]).T
    # data = scaler.transform(data)
    # pred_list = []
    # for i in range(4):
    #     pred_list.append(models[i].predict(data))
    # pred_list = np.vstack(pred_list).T
    # df = pd.DataFrame(pred_list, columns=header, index=m2)
    # df.to_csv(r'dataset/data_resultQ3Q4/sense_m2.csv')

    # # m3 分析
    # m3 = np.arange(39, 53, 0.5)
    # t1 = np.array([972]*m3.shape[0])
    # t2 = np.array([818]*m3.shape[0])
    # m1 = np.array([56]*m3.shape[0])
    # m2 = np.array([106]*m3.shape[0])
    # m4 = np.array([20]*m3.shape[0])
    # p3 = np.array([270]*m3.shape[0])
    # p4 = np.array([150]*m3.shape[0])
    # data = np.vstack([t1, t2, m1, m2, m3, m4, p3, p4]).T
    # data = scaler.transform(data)
    # pred_list = []
    # for i in range(4):
    #     pred_list.append(models[i].predict(data))
    # pred_list = np.vstack(pred_list).T
    # df = pd.DataFrame(pred_list, columns=header, index=m3)
    # df.to_csv(r'dataset/data_resultQ3Q4/sense_m3.csv')

    # # m4 分析
    # m4 = np.arange(15, 24, 0.4)
    # t1 = np.array([972]*m4.shape[0])
    # t2 = np.array([818]*m4.shape[0])
    # m1 = np.array([56]*m4.shape[0])
    # m2 = np.array([106]*m4.shape[0])
    # m3 = np.array([47]*m4.shape[0])
    # p3 = np.array([270]*m4.shape[0])
    # p4 = np.array([150]*m4.shape[0])
    # data = np.vstack([t1, t2, m1, m2, m3, m4, p3, p4]).T
    # data = scaler.transform(data)
    # pred_list = []
    # for i in range(4):
    #     pred_list.append(models[i].predict(data))
    # pred_list = np.vstack(pred_list).T
    # df = pd.DataFrame(pred_list, columns=header, index=m4)
    # df.to_csv(r'dataset/data_resultQ3Q4/sense_m4.csv')

    # # p3 分析
    # p3 = np.arange(210, 384, 6)
    # t1 = np.array([972]*p3.shape[0])
    # t2 = np.array([818]*p3.shape[0])
    # m1 = np.array([56]*p3.shape[0])
    # m2 = np.array([106]*p3.shape[0])
    # m3 = np.array([47]*p3.shape[0])
    # m4 = np.array([20]*p3.shape[0])
    # p4 = np.array([150]*p3.shape[0])
    # data = np.vstack([t1, t2, m1, m2, m3, m4, p3, p4]).T
    # data = scaler.transform(data)
    # pred_list = []
    # for i in range(4):
    #     pred_list.append(models[i].predict(data))
    # pred_list = np.vstack(pred_list).T
    # df = pd.DataFrame(pred_list, columns=header, index=p3)
    # df.to_csv(r'dataset/data_resultQ3Q4/sense_p3.csv')

    # p4 分析
    p4 = np.arange(110, 195, 3.5)
    t1 = np.array([972]*p4.shape[0])
    t2 = np.array([818]*p4.shape[0])
    m1 = np.array([56]*p4.shape[0])
    m2 = np.array([106]*p4.shape[0])
    m3 = np.array([47]*p4.shape[0])
    m4 = np.array([20]*p4.shape[0])
    p3 = np.array([270]*p4.shape[0])
    data = np.vstack([t1, t2, m1, m2, m3, m4, p3, p4]).T
    data = scaler.transform(data)
    pred_list = []
    for i in range(4):
        pred_list.append(models[i].predict(data))
    pred_list = np.vstack(pred_list).T
    df = pd.DataFrame(pred_list, columns=header, index=p4)
    df.to_csv(r'dataset/data_resultQ3Q4/sense_p4.csv')









