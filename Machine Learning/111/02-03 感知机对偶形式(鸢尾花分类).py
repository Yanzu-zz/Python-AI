from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

font = FontProperties(fname="C:\\Users\\IAdmin\\AppData\\Local\\Microsoft\\Windows\\Fonts\\DingTalk JinBuTi.ttf")


def get_data():
    df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    X = df.iloc[0:100, [0, 2]].values
    train_data_p = df.iloc[0:50, [0, 2, 4]].values
    train_data_n = df.iloc[50:100, [0, 2, 4]].values
    train_data_p[:, [2]], train_data_n[:[2]] = -1, 1
    train_data = train_data_p.tolist() + train_data_n.tolist()

    return train_data, X


def train(num_iter, train_data, learning_rate):
    w = 0.0
    b = 0
    data_length = len(train_data)
    alpha=[0 for _ in range(data_length)]
    gram=np.matmul(train_data[:,0:-1])






























