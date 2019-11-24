import matplotlib.pyplot as plt


def draw_single_variable(training_data, theta0, theta1):
    x_data = []
    y_data = []
    start_point, end_point = [0, 9], [theta0, theta0 + theta1 * 9]
    for (x, y) in training_data:
        x_data.append(x)
        y_data.append(y)
    plt.plot(x_data, y_data, 'ro')
    plt.plot(start_point, end_point)
    plt.show()
