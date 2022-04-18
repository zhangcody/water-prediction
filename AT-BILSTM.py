import scipy
from keras.layers import Input, Dense, LSTM, merge ,Conv1D,Dropout,Bidirectional,Multiply
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *

import matplotlib.pyplot as plt

import  pandas as pd
import  numpy as np



SINGLE_ATTENTION_VECTOR = False
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = inputs
    #a = Permute((2, 1))(inputs)
    #a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((1, 2), name='attention_vec')(a)

    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul

# 注意力机制的另一种写法 适合上述报错使用 来源:https://blog.csdn.net/uhauha2929/article/details/80733255
def attention_3d_block2(inputs, single_attention_vector=False):
    # 如果上一层是LSTM，需要return_sequences=True
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)
    # 乘上了attention权重，但是并没有求和，好像影响不大
    # 如果分类任务，进行Flatten展开就可以了
    # element-wise
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul



def create_dataset(dataset, look_back):
    '''
    对数据进行处理
    '''
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back),:]
        dataX.append(a)
        dataY.append(dataset[i + look_back,:])
    TrainX = np.array(dataX)
    Train_Y = np.array(dataY)

    return TrainX, Train_Y

#多维归一化  返回数据和最大最小值
def NormalizeMult(data):
    #normalize 用于反归一化
    data = np.array(data)
    normalize = np.arange(2*data.shape[1],dtype='float64')

    normalize = normalize.reshape(data.shape[1],2)
    print(normalize.shape)
    for i in range(0,data.shape[1]):
        #第i列
        list = data[:,i]
        listlow,listhigh =  np.percentile(list, [0, 100])
        # print(i)
        normalize[i,0] = listlow
        normalize[i,1] = listhigh
        delta = listhigh - listlow
        if delta != 0:
            #第j行
            for j in range(0,data.shape[0]):
                data[j,i]  =  (data[j,i] - listlow)/delta
    #np.save("./normalize.npy",normalize)
    return  data,normalize

#多维反归一化
def FNormalizeMult(data,normalize):
    data = np.array(data)
    for i in  range(0,data.shape[1]):
        listlow =  normalize[i,0]
        listhigh = normalize[i,1]
        delta = listhigh - listlow
        if delta != 0:
            #第j行
            for j in range(0,data.shape[0]):
                data[j,i]  =  data[j,i]*delta + listlow

    return data


def attention_model():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))

    x = Conv1D(filters = 64, kernel_size = 1, activation = 'tanh')(inputs)  #, padding = 'same'
    x = Dropout(0.02)(x)

    #lstm_out = Bidirectional(LSTM(lstm_units, activation='relu'), name='bilstm')(x)
    #对于GPU可以使用CuDNNLSTM
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    lstm_out = Dropout(0.02)(lstm_out)
    attention_mul = attention_3d_block2(lstm_out)
    attention_mul = Flatten()(attention_mul)

    output = Dense(1)(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model

def attention_model2():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))

    x = Conv1D(filters = 64, kernel_size = 1, activation = 'tanh')(inputs)  #, padding = 'same'
    x = Dropout(0.02)(x)

    #lstm_out = Bidirectional(LSTM(lstm_units, activation='relu'), name='bilstm')(x)
    #对于GPU可以使用CuDNNLSTM
    lstm_out = LSTM(lstm_units, return_sequences=True)(x)
    lstm_out = Dropout(0.02)(lstm_out)
    attention_mul = attention_3d_block2(lstm_out)
    attention_mul = Flatten()(attention_mul)

    output = Dense(1)(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model

#画出曲线变化图
def plot_curve(true_data, predicted):
    plt.figure(figsize=(12, 8))
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.plot(true_data, label='Actual',markersize='5',ls='--',lw='3',color = 'red')
    plt.plot(predicted, label='Predicted',markersize='5',ls='-',lw='3',color = 'blue')
    plt.legend(fontsize=30)
    # plt.plot(predicted_LSTM, label='Predicted data by LSTM') plt.legend()
    # plt.savefig('result_final.png')
    plt.show()

def data_split(data, train_len, lookback_window):
    train = data[:train_len]  #标志训练集
    test = data[train_len:]   #标志测试集
    # print(train.shape)

    #X1[]代表移动窗口中的10个数
    #Y1[]代表相应的移动窗口需要预测的数
    #X2, Y2 同理

    X1, Y1 = [], []
    for i in range(lookback_window, len(train)):
        X1.append(train[i - lookback_window:i])
        Y1.append(train[i])
    Y_train = np.array(Y1)
    X_train = np.array(X1)

    X2, Y2 = [], []
    for i in range(lookback_window, len(test)):
        X2.append(test[i - lookback_window:i])
        Y2.append(test[i])
    Y_test = np.array(Y2)
    X_test = np.array(X2)

    # print(X_train.shape)
    # print(Y_train.shape)
    return (X_train, Y_train, X_test, Y_test)

def RMSE(test, predicted):
    rmse = scipy.math.sqrt(mean_squared_error(test, predicted))
    return rmse



data = pd.read_csv('fuhe.csv')
data = data.drop(['time'], axis = 1)

print(data.columns)
print(data.shape)


INPUT_DIMS = 1
TIME_STEPS = 20
lstm_units = 64

#归一化
data,normalize = NormalizeMult(data)

X, Y = create_dataset(data,TIME_STEPS)

train_X = X[:int(len(data) * .80)]
test_X = X[int(len(data) * .80):]
train_Y = Y[:int(len(data) * .80)]
test_Y = Y[int(len(data) * .80):]

print(train_X.shape,train_Y.shape)

m = attention_model()
m.summary()
m.compile(optimizer=Adam(learning_rate=0.05), loss='mse')
m.fit([train_X], train_Y, epochs=60, batch_size=100, validation_split=0.1)

Y2_test_hat = m.predict(test_X)
prediction = FNormalizeMult(Y2_test_hat,normalize)
test = FNormalizeMult(test_Y,normalize)

plot_curve(test, prediction)

rmse = format(RMSE(test, prediction), '.4f')
r2 = format(r2_score(test, prediction), '.4f')
mae = format(mean_absolute_error(test, prediction), '.4f')
print('RMSE:' + str(rmse) + '\n' + 'MAE:' + str(mae) + '\n' + 'R2:' + str(r2))
