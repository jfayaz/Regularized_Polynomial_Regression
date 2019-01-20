function [theta] = trainLinearReg(X, y, lambda)
%Trains linear regression given a dataset (X, y) and a regularization parameter lambda
%   [theta] = trainLinearReg (X, y, lambda) trains linear regression using
%   the dataset (X, y) and regularization parameter lambda. Returns the
%   trained parameters theta.
%

% Initializing Theta
initial_theta = zeros(size(X, 2), 1); 

% Creating "short hand" for the cost function to be minimized
costFunction = @(t) linearRegCostFunction(X, y, t, lambda);

% Now, costFunction is a function that takes in only one argument
options = optimset('MaxIter', 200, 'GradObj', 'on');

% Minimizing using fmincg
theta = fmincg(costFunction, initial_theta, options);

end
