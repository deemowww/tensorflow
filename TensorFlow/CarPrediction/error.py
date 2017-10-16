import numpy as np


def checkList(predict, result):
    if len(predict) != len(result) or len(result) == 0:
        return False
    for i in result:
        if i == 0:
            return False
    return True


def MAPE(predict, result):
    if not checkList(predict, result):
        return np.NaN
    mape = 0
    for i in range(len(predict)):
        mape += abs((result[i][0] - predict[i][0]) / result[i])
    mape /= len(predict)
    return mape


def RMSPE(predict, result):
    if not checkList(predict, result):
        return np.NaN
    rmspe = 0
    for i in range(len(result)):
        rmspe += ((result[i] - predict[i]) / result[i]) ** 2
    rmspe /= len(result)
    rmspe = rmspe ** 0.5
    return rmspe


def RMSE(predict, result):
    if not checkList(predict, result):
        return np.NaN
    rmse = 0
    for i in range(len(result)):
        rmse += (result[i] - predict[i]) ** 2
    rmse /= len(result)
    rmse = rmse ** 0.5
    return rmse


def MAE(predict, result):
    if not checkList(predict, result):
        return np.NaN
    mae = 0
    for i in range(len(result)):
        mae += abs(result[i] - predict[i])
    mae /= len(result)
    return mae


def NMSE(predict, result):
    if not checkList(predict, result):
        return np.NaN
    up = 0
    down = 0
    mean = sum(result) / len(result)
    for i in range(len(result)):
        up += (result[i] - predict[i]) ** 2
        down += (result[i] - mean) ** 2
    nmse = up / down
    return nmse

if __name__ == '__main__':
    print(RMSPE([5,4,3,2,1],[4,3,2,1,1]))