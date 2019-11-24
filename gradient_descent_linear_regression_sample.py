import gradient_descent_linear_regression as gd

training_data = [(1, 2), (1.5, 2.5), (2.5, 4), (3, 3), (3, 4), (4, 3.9), (4.1, 5), (5, 4), (6, 5.9), (6, 5), (6.5, 7),
                 (7, 6), (8, 7), (8.1, 9), (5.5, 5), (6.5, 7), (7, 5), (7.5, 6), (8, 6.5)]

gd.train(training_data=training_data, alpha=0.006123)
