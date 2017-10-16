import numpy as np
import csv
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
import matplotlib.pyplot as plt
learn = tf.contrib.learn
import math
import os
import re
from TensorFlow.CarPrediction.error import *

def files():
    current = list(os.walk(os.path.dirname(os.path.realpath(__file__))))[1][2]
    allfiles = list(filter(lambda x: re.match(".*\\.csv", x), current))
    return allfiles


def train(FILE_NAME="本田CR-V.csv", HIDDEN_SIZE=20, NUM_LAYERS=2, TIMESTEPS=10, TRAINING_STEPS=3000, BATCH_SIZE=10, show=False):
    data1 = []
    with open("cardata/" + FILE_NAME, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            data1.append(int(row[-1]))


    data2 = data1[60:]
    data1 = data1[:60]

    data1 = np.array(data1)
    data2 = np.array(data2)


    def generate_data(seq):
        X = []
        y = []

        for i in range(len(seq) - TIMESTEPS):
            X.append([seq[i: i + TIMESTEPS]])
            y.append([seq[i + TIMESTEPS]])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


    def MaxMinNormalization(x1):
        minvalue = x1.min(0)
        maxvalue = x1.max(0)
        aver = (maxvalue - minvalue)/2.
        x = np.array([(i - minvalue) / (maxvalue - minvalue) for i in x1])
        return x


    def lstm_model(X, y):
        '''lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True)
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * NUM_LAYERS)

        output, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
        output = tf.reshape(output, [-1, HIDDEN_SIZE])'''
        stacked_rnn = []
        for i in range(NUM_LAYERS):
            stacked_rnn.append(tf.contrib.rnn.BasicLSTMCell(num_units=HIDDEN_SIZE, state_is_tuple=True))
        cell = tf.contrib.rnn.MultiRNNCell(stacked_rnn)

        output, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
        output = tf.reshape(output, [-1, HIDDEN_SIZE])

        # 通过无激活函数的全联接层计算线性回归，并将数据压缩成一维数组的结构。
        predictions = tf.contrib.layers.fully_connected(output, 1, None)

        # 将predictions和labels调整统一的shape
        labels = tf.reshape(y, [-1])
        predictions = tf.reshape(predictions, [-1])

        loss = tf.losses.mean_squared_error(predictions, labels)

        train_op = tf.contrib.layers.optimize_loss(
            loss, tf.contrib.framework.get_global_step(),
            optimizer="Adagrad", learning_rate=0.1)

        return predictions, loss, train_op


    # 封装之前定义的lstm。
    regressor = SKCompat(learn.Estimator(model_fn=lstm_model, model_dir=""))


    train_X, train_y = generate_data(data1)

    test_X, test_y = generate_data(data2)
    maxvalue = data2.max(0)
    train_X = MaxMinNormalization(train_X)
    train_y = MaxMinNormalization(train_y)
    test_X = MaxMinNormalization(test_X)

    # 拟合数据。
    regressor.fit(train_X, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)

    # 计算预测值。
    predicted = [[pred * maxvalue] for pred in regressor.predict(test_X)]
    # 计算MSE。
    rmse = np.sqrt(((predicted - test_y) ** 2).mean(axis=0))
    if show:
        print("MAPE is: %f" % MAPE(predicted, test_y))
        print("RMSPE is: %f" % RMSPE(predicted, test_y))
        print("RMSE is: %f" % RMSE(predicted, test_y))
        print("MAE is: %f" % MAE(predicted, test_y))
        print("NMSE is: %f" % NMSE(predicted, test_y))
        plot_predicted, = plt.plot(predicted, label='predicted')
        plot_test, = plt.plot(test_y, label='real')

        plt.legend([plot_predicted, plot_test],['predicted', 'real'])
        plt.show()
    else:
        return [MAPE(predicted, test_y)[0], RMSPE(predicted, test_y)[0]]


if __name__ == '__main__':
    if False:
        with open("train_data/data.csv", "w", newline='', encoding="utf-8") as f:
            csv_writer = csv.writer(f, dialect='excel')
            csv_writer.writerow(['车型', "MAPE", "RMSPE"])
            for filename in files():
                car_data = [filename.split('.')[0]]
                train_data = train(FILE_NAME=filename)
                car_data.extend(train_data)
                csv_writer.writerow(car_data)
        '''
        train(show=True)'''
    else:
        train(show=True)



