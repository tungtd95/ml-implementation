training_data_single_variable = [[1, 2],
                                 [1.5, 2.5],
                                 [2.5, 4],
                                 [3, 3],
                                 [3, 4],
                                 [4, 3.9],
                                 [4.1, 5],
                                 [5, 4],
                                 [6, 5.9],
                                 [6, 5],
                                 [6.5, 7],
                                 [7, 6],
                                 [8, 7],
                                 [8.1, 9],
                                 [5.5, 5],
                                 [6.5, 7],
                                 [7, 5],
                                 [7.5, 6],
                                 [8, 6.5]];

raw_x = training_data_single_variable(:, 1);
m = size(training_data_single_variable, 1);
n = size(training_data_single_variable, 2);
% generate vector x
x = [ones(m, 1) raw_x];
% generate vector y
y = training_data_single_variable(:, 2);
% generate vector theta
theta = rand(n, 1);
alpha = 0.006123;
while true
   theta_temp = theta - (alpha/m) * x' * ((x * theta) - y);
   if theta_temp(1:1,1:1) - theta(1:1,1:1) < 0.000001
        break;
   end
   theta = theta_temp;
end
disp(theta);
plot(raw_x, y, 'ro');
hold on;
plot([min(raw_x), max(raw_x)], [theta(1, 1) + theta(2, 1) * min(raw_x), theta(1, 1) + theta(2, 1) * max(raw_x)])
pause;
