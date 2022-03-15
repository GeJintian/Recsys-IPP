%load('data.mat');
[num_hotels,num_users] = size(Y); 
num_features = 3;

%Normalization
[Ynorm, Ymean] = normalizeRatings(Y, R);

%Initialize X and Theta
X = randn(num_hotels,num_features);
Theta = randn(num_users,num_features);
initial_parameters = [X(:);Theta(:)];

%Use fmincg to train theta
options = optimset('GradObj','on','MaxIter',100);
lambda = 1;
theta = fmincg(@(t)(Costfunction(t, Ynorm, R, num_users, num_hotels, num_features,lambda)), initial_parameters, options);

% Unfold the returned theta back into X and Theta
X = reshape(theta(1:num_hotels*num_features), num_hotels, num_features);
Theta = reshape(theta(num_hotels*num_features+1:end), num_users, num_features);

%============Training is completed==============

%Recommandations
% P_total = X*Theta'+Ymean;
% P_sub = [];
% for i = 1:size(user_sub)
%     k = user_sub(i);
%     p = P_total(:,k);
%     P_sub = [P_sub,p];
% end
