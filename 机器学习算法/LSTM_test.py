import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tushare as ts
import talib
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.preprocessing.sequence import TimeseriesGenerator

class LSTM_Predict:
    stock_code = ''
    tsData = pd.DataFrame()

    def __init__(self, stock_code):
        self.stock_code = stock_code
    def date_setting(self, start_date, end_date):
        ts.set_token('fe3907f6ea74fe750f900a6964d17c5b8af1cce1d58c987665842109')
        #ts = ts.pro_api('fe3907f6ea74fe750f900a6964d17c5b8af1cce1d58c987665842109')
        self.tsData = ts.get_hist_data(code=self.stock_code, start=start_date, end=end_date)
        self.tsData = self.tsData.sort_index(ascending=True).reset_index()
        #print(self.tsData)
    def makePrediction(self, node):
        # 创建数据框
        new_data = pd.DataFrame(index=range(0, len(self.tsData)), columns=['Date', 'Open', 'High', 'Low', 'Close','upper_band','middle_band','lower_band','up_down'])
        # 计算bollinger bands
        upper_band, middle_band, lower_band = talib.BBANDS(self.tsData["close"], timeperiod=20)
        # 计算macd
        #macd, signal, hist = talib.MACD(self.tsData["close"], fastperiod=12, slowperiod=26, signalperiod=9)
        for i in range(0, len(self.tsData)):
            new_data['Date'][i] = self.tsData.index[i]
            new_data['Open'][i] = self.tsData["open"][i]
            new_data['High'][i] = self.tsData["high"][i]
            new_data['Close'][i] = self.tsData["close"][i]
            new_data['Low'][i] = self.tsData["low"][i]
            new_data['upper_band'][i] = round(upper_band[i],2)
            new_data['middle_band'][i] = round(middle_band[i],2)
            new_data['lower_band'][i] = round(lower_band[i],2)
            if(i<(len(self.tsData)-1)):
                if(self.tsData["p_change"][i+1]>0):
                    new_data['up_down'][i] = 1
                elif(self.tsData["p_change"][i+1]<0):
                    new_data['up_down'][i] = -1
                else:
                    new_data['up_down'][i] = 0
        
        #print(self.tsData["close"])
        #print(new_data)

        # 设置索引
        new_data.index = new_data.Date
        new_data.drop('Date', axis=1, inplace=True)

        # 创建训练集和验证集
        dataset = new_data.values
        #print(dataset)
        #exit()
        train = dataset[0:node, :]
        #x_train = dataset[0:node, :-1]
        #y_train = dataset[0:node, -1]
        x_train = dataset[-5:-1, :-1]
        y_train = dataset[-5:-1, -1]
        x_train = x_train.astype(float)  # numpy强制类型转换
        #x_train = x_train.astype(np.float)
        y_train = y_train.astype(float)  # numpy强制类型转换
        #y_train = y_train.astype(np.float)
        #a = a.astype(np.float)
        #print(x_train)
        #print(y_train)
        #print(x_train.shape)
        #print(y_train.shape)
        #exit()
        valid = dataset[node:, :]
        """ x_train = [[12.55, 12.69, 12.46, 12.67, 13.49, 12.87, 12.25],
           [12.64, 12.72, 12.59, 12.65, 13.32, 12.82, 12.32],
           [12.64, 12.72, 12.59, 12.65, 13.32, 12.82, 12.32],
           [12.6, 12.61, 12.5, 12.58, 13.17, 12.77, 12.38]]
        y_train = [-1, -1, 1, 1]  """
        # 将训练数据集转换为3D张量
        x_train, y_train = np.array(x_train), np.array(y_train)
        y_train = y_train.reshape(-1, 1)
        #x_train = torch.from_numpy(x_train).float()
        #y_train = torch.from_numpy(y_train).float()
        
        
        #print(x_train)
        #print(x_train.shape)
        #print(y_train)
        #print(y_train.shape)
        #exit()

        # 将数据集转换为x_train和y_train
        #scaler = MinMaxScaler(feature_range=(0, 1))
        #scaled_data = scaler.fit_transform(dataset)
        #x_train = scaler.fit_transform(x_train)
        #y_train = scaler.fit_transform(y_train)
        # 创建训练数据集
        """ x_train, y_train = [], []
        for i in range(60, len(train)):
            # 创建包含过去60天数据的数据集
            x_train.append(scaled_data[i - 60:i, 0])
            # 创建包含第61天数据的标签集
            y_train.append(scaled_data[i, 0]) """

        x_train = np.reshape(x_train, (x_train.shape[0],1,x_train.shape[1]))  # 将形状从 (130, 7) 转换为 (130, 7, 1)
        #x_train = np.reshape(x_train, (x_train.shape[1], 1))
        #x_train = x_train.reshape(x_train.shape[0], 1,x_train.shape[1])
        #y_train = y_train.reshape(-1, 1)

        print(x_train,y_train,x_train.shape,y_train.shape)
        #exit()


        """ # 创建和拟合LSTM网络
        model = Sequential()
        # 添加第一层LSTM和输入层
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        # 添加第二层LSTM
        model.add(LSTM(units=50))
        # 添加输出层
        model.add(Dense(1))
        # 编译模型
        model.compile(loss='mean_squared_error', optimizer='adam')
        # 训练模型
        model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2) """

         #机器人推荐的（投资成功、投资失败和投资打平三种类别）模型

        # 构建模型
        model = Sequential()
        model.add(LSTM(512, input_shape=(x_train.shape[1],x_train.shape[2]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(512))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='softmax'))

        # 编译模型
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy']) 

        # 训练模型
        model.fit(x_train, y_train, epochs=10, batch_size=10, verbose=2)

        exit()

        """ #机器人推荐的（涨，跌）模型
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout

        # 构建模型
        model = Sequential()
        model.add(LSTM(512, input_shape=(timesteps, input_dim), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(512))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        # 编译模型
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        # 训练模型
        model.fit(data, labels, epochs=10, batch_size=64, verbose=2) """

        """ #机器人推荐的数据结构编写
        import numpy as np

        # 定义数据结构
        data = np.zeros((num_bars, 6))
        # 前四列为开盘价、最高价、最低价和收盘价
        data[:, 0] = open_prices
        data[:, 1] = high_prices
        data[:, 2] = low_prices
        data[:, 3] = close_prices
        # 第五列为boll数据
        data[:, 4] = boll_data
        # 第六列为macd数据
        data[:, 5] = macd_data

        # 定义标签
        labels = np.zeros((num_bars, 1))
        # 第二列为下一个bar的涨跌情况，1表示涨，0表示跌
        labels[:, 0] = up_down_labels 

        # 将标签转换为one-hot编码
        from tensorflow.keras.utils import to_categorical
        one_hot_labels = to_categorical(labels)

        # 训练模型
        model.fit(data, one_hot_labels, epochs=10, batch_size=64, verbose=2)
        """


        # 使用过去值来预测
        inputs = new_data[len(new_data) - len(valid) - 60:].values
        inputs = inputs.reshape(-1, 1)
        inputs = scaler.transform(inputs)
        X_test = []
        for i in range(60, inputs.shape[0]):
            X_test.append(inputs[i - 60:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        closing_price = model.predict(X_test)
        closing_price = scaler.inverse_transform(closing_price)

        # 作图
        train = new_data[:node]
        valid = new_data[node:]
        print('valid长度是：' + str(len(valid)))
        print(len(closing_price))
        valid['Predictions'] = closing_price
        plt.plot(train['Close'], label='训练集')
        plt.plot(valid['Close'], label='真实值')
        plt.plot(valid['Predictions'], label='预测值')
        plt.show()

    def print(self):
        print(self.tsData)

a = LSTM_Predict('000001')
a.date_setting(start_date='2022-04-01', end_date='2023-04-10')
a.makePrediction(130)