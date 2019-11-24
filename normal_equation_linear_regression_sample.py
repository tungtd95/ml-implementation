import normal_equation_linear_regression as ne

training_data_single_variable = [[10, 2],
                                 [15, 2.5],
                                 [25, 4],
                                 [30, 3],
                                 [30, 4],
                                 [40, 3.9],
                                 [41, 5],
                                 [50, 4],
                                 [60, 5.9],
                                 [60, 5],
                                 [65, 7],
                                 [70, 6],
                                 [80, 7],
                                 [81, 9],
                                 [55, 5],
                                 [65, 7],
                                 [70, 5],
                                 [75, 6],
                                 [80, 6.5]]

ne.train(training_data_single_variable)
