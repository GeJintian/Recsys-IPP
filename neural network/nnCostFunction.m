function [J,grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, X, y, n, lambda)
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

m = size(X, 1); % m is the number of training set(hotel)

K = n; % Number of output units

A1 = [ones(m, 1), X];
Z2 = A1 * Theta1';
A2 = [ones(m,1),1./(1+exp(-1*Z2))];
Z3 = A2 * Theta2';
A3 = 1./(1+exp(-1*Z3));
for j = 1:m
    for k = 1:K
        J = J - y(j,k) * log(A3(j,k)) - (1-y(j,k)) * log(1-A3(j,k));
    end
end
J = 1/m * J;
T1 = Theta1;
T1(:, 1) = 0;
T2 = Theta2;
T2(:, 1) = 0;
J = J + lambda/(2 * m) * (sum(sum(T1.^2)) + sum(sum(T2.^2)));
Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));
for i = 1:m
    a1 = X(i,:)';
    a1 = [1;a1];
    z2 = Theta1 * a1;
    a2 = [1;sigmoid(z2)];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    
    delta3 = a3 - y(i,:)';
    delta2 = Theta2' * delta3;
    delta2 = delta2(2:end);
    delta2 = delta2 .* sigmoidGradient(z2);
    Delta1 = Delta1 + delta2 * a1';
    Delta2 = Delta2 + delta3 * a2';
end

Theta1_grad = 1/m * Delta1;
Theta2_grad = 1/m * Delta2;


% Unroll gradients
grad = [Theta1_grad(:); Theta2_grad(:)];
end

