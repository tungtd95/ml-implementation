import matplotlib.pyplot as plt


def train(training_data, alpha):
    # init some random value for theta
    theta0 = 27
    theta1 = 7

    m = len(training_data)
    print("start training...")
    while True:
        theta0_temp = theta0 - alpha * _cal_derivative_partial_theta_0(training_data, theta0, theta1) / m
        theta1_temp = theta1 - alpha * _cal_derivative_partial_theta_1(training_data, theta0, theta1) / m

        # stop training when converged
        if theta0_temp == theta0 or theta1_temp == theta1:
            break
        theta0 = theta0_temp
        theta1 = theta1_temp
    print("result: (theta0, theta1) = ({t1}, {t2})".format(t1=theta0, t2=theta1))
    draw(training_data, theta0, theta1)


# calculate the partial derivative of theta0
def _cal_derivative_partial_theta_0(training_data, theta0, theta1):
    sum_derivative = 0
    for (x, y) in training_data:
        sum_derivative = sum_derivative + (theta0 + theta1 * x - y)
    return sum_derivative


# calculate the partial derivative of theta1
def _cal_derivative_partial_theta_1(training_data, theta0, theta1):
    sum_derivative = 0
    for (x, y) in training_data:
        sum_derivative = sum_derivative + (theta0 + theta1 * x - y) * x
    return sum_derivative


def draw(training_data, theta0, theta1):
    x_data = []
    y_data = []
    start_point, end_point = [0, 9], [theta0, theta0 + theta1 * 9]
    for (x, y) in training_data:
        x_data.append(x)
        y_data.append(y)
    plt.plot(x_data, y_data, 'ro')
    plt.plot(start_point, end_point)
    plt.show()
