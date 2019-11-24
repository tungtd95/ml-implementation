import gradient_descent_multiple_variable as gd

training_data_single_variable = [(1, 2), (1.5, 2.5), (2.5, 4), (3, 3), (3, 4), (4, 3.9), (4.1, 5),
                                 (5, 4), (6, 5.9), (6, 5), (6.5, 7), (7, 6), (8, 7), (8.1, 9), (5.5, 5),
                                 (6.5, 7), (7, 5), (7.5, 6), (8, 6.5)]

alpha = 0.006123
# for comparing result with gradient_descent_linear_regression
# gd.train(training_data_single_variable, alpha)

training_data_multiple_variable = [(1, 2, 3), (1.5, 2.5, 4), (2.5, 4, 4), (3, 3, 3.5), (3, 4, 5), (4, 3.9, 5.5),
                                   (4.1, 5, 6), (5, 4, 7), (6, 5.9, 8), (6, 5, 7.5), (6.5, 7, 8.5), (7, 6, 9),
                                   (8, 7, 9), (8.1, 9, 11), (5.5, 5, 7), (6.5, 7, 8), (7, 5, 7.5), (7.5, 6, 8),
                                   (8, 6.5, 9)]

# for demo of multiple variable implementation
gd.train(training_data_multiple_variable, alpha)
