import pandas as pd
from sklearn.preprocessing import scale
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Z-Score 标准化
# df = pd.read_csv(r'dataset/result.csv')
# cur_data = df[df['index A'].notnull()]
# df['Temperature of system I'] = scale(df['Temperature of system I'])
# df['Temperature of system II'] = scale(df['Temperature of system II'])
# df['Mineral_parameter_1'] = scale(df['Mineral_parameter_1'])
# df['Mineral_parameter_2'] = scale(df['Mineral_parameter_2'])
# df['Mineral_parameter_3'] = scale(df['Mineral_parameter_3'])
# df['Mineral_parameter_4'] = scale(df['Mineral_parameter_4'])
# df.to_csv('dataset/scale_result.csv', index=False)


def get_tar_idx(notnull_idx, k):
    """
    根据非空元素，选择最近邻的前后k个元素作为预测目标
    :param notnull_idx: 非空元素的index集合
    :param k: 对每个非空元素，前后各取多少元素作为预测目标
    :return:
    """
    total = set([])
    for idx in notnull_idx:
        tmp = np.arange(idx-k, idx+k+1, 1)
        tmp[(tmp < 0)] = 1  # 超出数据集的部分
        tmp[(tmp > 2879)] = 2879  # 超出数据集的部分, Q1Q2  Q3Q4的值不同，记得调
        total.update(tmp)
    total = list(total)
    return np.array([elm for elm in total if elm not in notnull_idx])  # 剔除训练集部分


def split_data(data):
    """
    将数据集划分为输入数据与标签数据
    :param data:
    :return:
    """
    t1, t2 = data['Temperature of system I'].to_numpy().reshape(-1), data['Temperature of system II'].to_numpy().reshape(-1)
    m1, m2, m3, m4 = data['Mineral_parameter_1'].to_numpy().reshape(-1), data['Mineral_parameter_2'].to_numpy().reshape(-1), data['Mineral_parameter_3'].to_numpy().reshape(-1), data['Mineral_parameter_4'].to_numpy().reshape(-1)
    A, B, C, D = data['index A'].to_numpy().reshape(-1), data['index B'].to_numpy().reshape(-1), data['index C'].to_numpy().reshape(-1), data['index D'].to_numpy().reshape(-1)
    data = np.array([t1, t2, m1, m2, m3, m4]).T
    labels = np.array([A, B, C, D]).T
    return data, labels


def get_TFmodel(layer=20, hidden=64):
    """

    :param layer:
    :param hidden:
    :return:
    """
    model = Sequential([])
    for _ in range(layer):
        model.add(Dense(hidden, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model


def get_XGBmodel():
    return xgb.XGBRegressor()


def get_Linearmodel():
    return LinearRegression()


def calc_loss(y_true, y_pred):
    """
    计算均方误差
    :param y_true:
    :param y_pred:
    :return:
    """
    return float(sum([(y1-y2)**2 for y1, y2 in zip(y_true, y_pred)])/y_true.shape[0])


if __name__ == '__main__':
    pass
    """
    第一问
    选用DNN、XGBoost、LinearModel进行插值，每次插值之前，统计训练集上的损失，并记录.最后选择损失最小的方法  使用文件scale_result.csv
    """
    df = pd.read_csv(r'dataset/scale_result.csv')
    header = ['index A', 'index B', 'index C', 'index D']
    iteration = ['one', 'two', 'three', 'four', 'five', 'six']
    k_list = [1, 3, 5, 7, 9, 11]
    loss_list = []  # 统计损失，方便制表
    for i in range(6):  # 做6次迭代，可以填满大部分空值
        train_index = df['index A'].notnull()
        train_data = df[train_index]
        train_index = np.argwhere(np.int32(train_index) == 1).reshape(-1)  # 转成了numpy 方便找前后的target数据
        tar_idx = get_tar_idx(train_index, k=k_list[i])
        tar_data = df.loc[tar_idx, :]
        train_data, train_labels = split_data(train_data)
        tar_data, _ = split_data(tar_data)
        for k in range(4):  # 4个index
            """
            如果换模型，只需修改3个地方：model = get_TFmodel()   model.fit(x_train, y_train, epochs=100, verbose=0)  与最后的保存位置
            """
            model = get_Linearmodel()
            x_train = train_data  # 为了好看点。。
            y_train = train_labels[:, k].reshape(-1)
            model.fit(x_train, y_train)  # shut up!
            y_pred = model.predict(x_train)
            train_loss = calc_loss(y_train, y_pred)
            print('第{}次迭代，模型在{}上的损失为：{}'.format(i+1, header[k], train_loss))
            loss_list.append(train_loss)
            predict = model.predict(tar_data).reshape(-1)
            assert tar_idx.shape[0] == predict.shape[0]
            for j in range(tar_idx.shape[0]):
                df[header[k]][tar_idx[j]] = predict[j]

    # 最后一次插值，对象为所有空值 顺带把第一题做了
    train_index = df['index A'].notnull()
    train_data = df[train_index]
    tar_idx = df['index A'].isnull()
    tar_data = df[tar_idx]
    tar_idx = np.argwhere(np.int32(tar_idx) == 1).reshape(-1)  # 这里==1 实际找的是空值，因为上面使用的是isnull() 很烦
    train_data, train_labels = split_data(train_data)
    tar_data, _ = split_data(tar_data)
    for k in range(4):
        model = get_Linearmodel()
        x_train = train_data  # 为了好看点。。
        y_train = train_labels[:, k].reshape(-1)
        model.fit(x_train, y_train)  # shut up!
        predict = model.predict(tar_data).reshape(-1)
        assert tar_idx.shape[0] == predict.shape[0]
        for j in range(tar_idx.shape[0]):
            df[header[k]][tar_idx[j]] = predict[j]
    #
    # loss_list = np.array(loss_list)
    # loss_list = loss_list.reshape((len(iteration), len(header)))
    # loss_tabel = pd.DataFrame(loss_list, index=iteration, columns=header)
    # loss_tabel.to_csv('dataset/实验结果/Linear_loss.csv', index=False)
    # df.to_csv('dataset/实验结果/Linear_result.csv', index=False)
    # ———————————— #

    # 模型验证，用插好的值训练，预测已有的真实值 使用文件Q1test.csv
    # df1 = pd.read_csv(r'dataset/result.csv')
    # tar_idx = df1['index A'].notnull()
    # train_idx = ~tar_idx  # Magic! ~取反，注意如果dtype为object可能会报错
    # df = pd.read_csv(r'dataset/实验结果/Q1test.csv')
    # train_data = df[train_idx]
    # tar_data = df[tar_idx]
    # x_train, y_train = split_data(train_data)
    # x_test, y_test = split_data(tar_data)
    # for i in range(4):
    #     model = get_XGBmodel()
    #     model.fit(x_train, y_train[:, i].reshape(-1))
    #     y_pred = model.predict(x_test)
    #     loss = calc_loss(y_test[:, i].reshape(-1), y_pred)
    #     print(loss)

    # """
    # 第二问
    # 使用由XGBoost进行插值的数据。并把第二问的数据复制在了最底端 使用Q2.csv
    # """
    # df = pd.read_csv(r'dataset/实验结果/Q2.csv')
    # train_idx = df['Temperature of system I'].notnull()
    # tar_idx = ~train_idx
    # train_data = df.loc[train_idx, ('Temperature of system I', 'Temperature of system II', 'index A', 'index B', 'index C', 'index D', 'Mineral_parameter_1', 'Mineral_parameter_2', 'Mineral_parameter_3', 'Mineral_parameter_4')].to_numpy()
    # tar_data = df.loc[tar_idx, ('Temperature of system I', 'Temperature of system II', 'index A', 'index B', 'index C', 'index D', 'Mineral_parameter_1', 'Mineral_parameter_2', 'Mineral_parameter_3', 'Mineral_parameter_4')].to_numpy()
    # tar_data[np.isnan(tar_data)] = 0  # 暂时用0代替空值，方便后续的标准化
    # scaler = StandardScaler()
    # train_data = scaler.fit_transform(train_data)  # StandardScaler没有axis操作，只能按照(samples, features)的形式输入数据
    # tar_data = scaler.transform(tar_data)
    # x_train, y_train = train_data[:, 2:], train_data[:, 0:2]
    # x_test, y_test = tar_data[:, 2:], tar_data[:, 0:2]  # 这里的y_test是无用的，需要后续计算替代
    # for i in range(2):
    #     model = get_XGBmodel()
    #     model.fit(x_train, y_train[:, i])
    #     y_pred = model.predict(x_test).reshape((-1))
    #     y_test[:, i] = y_pred  # 替代原有标签
    # final_data = np.hstack([y_test, x_test])
    # print(scaler.inverse_transform(final_data))

















