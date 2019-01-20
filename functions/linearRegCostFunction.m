function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%Compute cost and gradient for regularized linear regression with multiple variables
%   [J, grad] = linearRegCostFunction(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initializing
m = length(y);              % number of training examples
J = 0;
grad = zeros(size(theta));

% Calculating cost function
diff = X*theta - y;

% Calculating penalty
% excluded the first theta value
theta1 = [0 ; theta(2:end, :)];
p = lambda*(theta1'*theta1);
J = (diff'*diff)/(2*m) + p/(2*m);

% Calculating grads
grad = (X'*diff+lambda*theta1)/m;

end
