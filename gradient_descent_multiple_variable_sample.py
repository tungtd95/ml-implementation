import gradient_descent_multiple_variable as gd

training_data_single_variable = [[1, 2], [1.5, 2.5], [2.5, 4], [3, 3], [3, 4], [4, 3.9], [4.1, 5],
                                 [5, 4], [6, 5.9], [6, 5], [6.5, 7], [7, 6], [8, 7], [8.1, 9], [5.5, 5],
                                 [6.5, 7], [7, 5], [7.5, 6], [8, 6.5]]

alpha = 0.0006123
# for comparing result with gradient_descent_linear_regression
'''
in case alpha = 0.0006123 and training data like this:
training_data_single_variable = [[1, 2], [1.5, 2.5], [2.5, 4], [3, 3], [3, 4], [4, 3.9], [4.1, 5],
                                 [5, 4], [6, 5.9], [6, 5], [6.5, 7], [7, 6], [8, 7], [8.1, 9], [5.5, 5],
                                 [6.5, 7], [7, 5], [7.5, 6], [8, 6.5]]
when turn Feature Scaling on, J(theta) take 3704 milliseconds to converge
when turn Feature Scaling off, J(theta) take 23379 milliseconds to converge

* The Feature Scaling and the Learning Rate threshold does not effect anything but reducing training time
'''
gd.train(training_data_single_variable, alpha, should_scale=True)

training_data_multiple_variable = [[10, 0.2, 3],
                                   [15, 0.25, 4],
                                   [25, 0.4, 4],
                                   [30, 0.3, 3.5],
                                   [30, 0.4, 5],
                                   [40, 0.39, 5.5],
                                   [41, 0.5, 6],
                                   [50, 0.4, 7],
                                   [60, 0.59, 8],
                                   [60, 0.5, 7.5],
                                   [65, 0.7, 8.5],
                                   [70, 0.6, 9],
                                   [80, 0.7, 9],
                                   [81, 0.9, 11],
                                   [55, 0.5, 7],
                                   [65, 0.7, 8],
                                   [70, 0.5, 7.5],
                                   [75, 0.6, 8],
                                   [80, 0.65, 9]]

# for demo of multiple variable implementation
'''
in case alpha = 0.0006123 and training data like this:
training_data_multiple_variable = [[10, 0.2, 3],
                                   [15, 0.25, 4],
                                   [25, 0.4, 4],
                                   [30, 0.3, 3.5],
                                   [30, 0.4, 5],
                                   [40, 0.39, 5.5],
                                   [41, 0.5, 6],
                                   [50, 0.4, 7],
                                   [60, 0.59, 8],
                                   [60, 0.5, 7.5],
                                   [65, 0.7, 8.5],
                                   [70, 0.6, 9],
                                   [80, 0.7, 9],
                                   [81, 0.9, 11],
                                   [55, 0.5, 7],
                                   [65, 0.7, 8],
                                   [70, 0.5, 7.5],
                                   [75, 0.6, 8],
                                   [80, 0.65, 9]]
when turn Feature Scaling on, J(theta) take 5973 milliseconds to converge
when turn Feature Scaling off, J(theta) take 27504 milliseconds to converge
'''
# gd.train(training_data_multiple_variable, alpha)
