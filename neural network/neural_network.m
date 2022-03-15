function [Theta1, Theta2] = neural_network(Theta1, Theta2, X, y)
%Requires: X should be m * 2449 size.
%          Y should be m * n size.
%          m is the number of hotels, n is the output units


% Building first nn for hotel
n = 25;

% Building layers
input_layer_size  = 2449;  % 2449 input units
hidden_layer_size = 100;    % 100 hidden units

% Unroll parameters
initial_nn_params = [Theta1(:) ; Theta2(:)];

options = optimset('MaxIter', 50);
lambda = 0.1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, ~] = fmincg(costFunction, initial_nn_params, options);
% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));
end
