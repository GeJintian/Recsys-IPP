

% Building first nn for hotel
n = 25;
p = 100; %Data has been divided into 100 parts

% Building layers
input_layer_size  = 2449;  % 2449 input units
hidden_layer_size = 100;    % 100 hidden units

% Random initialize Theta
Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
Theta2 = randInitializeWeights(hidden_layer_size, n);

% Train Theta
for i = 1 : p
    load();% load training data
    [Theta1, Theta2] = neural_network(Theta1, Theta2, X, y);
end

% Prediction
load();% load submit data, data  saved in T, impression saved in IMP
rate = string(zeros(size(T,1),1));
A = 1:25;
pred = zeros(1,25);
ratemat = zeros(25,1);
for j = 1 : size(T, 1)
    imp = IMP(:,j);
    data = [1, T(j,:)];
    z1 = data * Theta1';
    a1 = [1, z1];
    score = a1 * Theta2';
    sort_score = score(pred);
    for i = 1:25
        pred(i) = A(find(score == sort_score(26-i)));
    end
    for i = 1:25
        ratemat(i,1) = imp(pred(i),1);
    end
    rate(j) = join(string(ratemat(:, 1)')," ");
end
submission = array2table(rate, 'VariableNames', {'item_recommendations'});
