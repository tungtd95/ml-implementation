import numpy as np
import random
import plot_helper as plt


def train(training_data, alpha):
    # pre-process raw data
    m = len(training_data)
    n = len(training_data[0])

    # initiate random theta vector
    theta_array = []
    for i in range(n):
        theta_array.append(random.randint(0, 100))
    theta_mtx = np.array(theta_array).reshape(n, 1)

    # filter x and y from raw data
    x_array = []
    y_array = []
    for data_item in training_data:
        x_array_item = [1]
        for i in range(n - 1):
            x_array_item.append(data_item[i])
        x_array.append(x_array_item)
        y_array.append(data_item[-1])
    x_mtx = np.array(x_array)
    y_mtx = np.array(y_array)
    print("start training...")
    while True:
        theta_temp = []
        for j in range(n):
            theta_temp.append(_cal_theta_element(theta_mtx, theta_mtx[j], x_mtx, y_mtx, alpha, m, n, j))
        theta_temp_mtx = np.array(theta_temp).reshape(n, 1)
        if np.array_equal(theta_mtx, theta_temp_mtx):
            break
        theta_mtx = theta_temp_mtx
    print(theta_mtx)
    if n == 2:
        plt.draw_single_variable(training_data, theta_mtx[0], theta_mtx[1])


def _cal_theta_element(theta_mtx, theta, x_mtx, y_mtx, alpha, m, n, index_j):
    theta_temp = theta - alpha * _cal_partial_derivative(theta_mtx, x_mtx, y_mtx, m, n, index_j) / m
    return theta_temp


def _cal_partial_derivative(theta_mtx, x_mtx, y_mtx, m, n, index_j):
    s = 0
    for i in range(m):
        s = s + (np.matmul(theta_mtx.reshape(1, n), x_mtx[i].reshape(n, 1))[0] - y_mtx[i]) * x_mtx[i][index_j]
    return s


