import matplotlib.pyplot as plt


def draw_single_variable(training_data, theta0, theta1):
    x_data = []
    y_data = []
    for (x, y) in training_data:
        x_data.append(x)
        y_data.append(y)
    min_x = min(x_data)
    max_x = max(x_data)
    start_point, end_point = [min_x, max_x], [theta0 + theta1 * min_x, theta0 + theta1 * max_x]
    plt.plot(x_data, y_data, 'ro')
    plt.plot(start_point, end_point)
    plt.show()
