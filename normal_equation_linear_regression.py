import numpy as np
import time
import plot_helper as plt


def train(training_data):
    t0 = int(round(time.time() * 1000))
    n = len(training_data[0])
    m = len(training_data)
    x = []
    y = []
    for element in training_data:
        x_array = [1]
        for i in range(n - 1):
            x_array.append(element[i])
        x.append(x_array)
        y.append(element[-1])
    x_mtx = np.array(x)
    y_mtx = np.array(y)
    x_mtx_trans = x_mtx.reshape((n, m))
    x_mtx_trans_mul_x_mtx = np.linalg.pinv(np.matmul(x_mtx_trans, x_mtx))
    first_operation = np.matmul(x_mtx_trans_mul_x_mtx, x_mtx_trans)
    theta = np.matmul(first_operation, y_mtx)
    t1 = int(round(time.time() * 1000))
    print("training time = {time} milliseconds".format(time=(t1 - t0)))
    print(theta)
    if n == 2:
        plt.draw_single_variable(training_data, theta[0], theta[1])
