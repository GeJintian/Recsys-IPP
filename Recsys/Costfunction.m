function [J,grad] = Costfunction(params, Y, R, num_users, num_hotels, num_features, lambda)
%Unfold X and Theta matrices from params
X = reshape(params(1:num_hotels*num_features), num_hotels, num_features);
Theta = reshape(params(num_users*num_features+1:end), ...
                num_users, num_features);
%Computing costfunction
M = X*Theta'-Y;
J = 1/2*sum(sum((R.*M).^2));
%Computing gradient
X_grad = (R.*M)*Theta;
Theta_grad = (X'*(R.*M))';
%Regularization
J = J + lambda/2*sum(sum(Theta.^2))+lambda/2*sum(sum(X.^2));
X_grad = X_grad + lambda*X;
Theta_grad = Theta_grad + lambda*Theta;
%Fold grad
grad = [X_grad(:);Theta_grad(:)];
end

