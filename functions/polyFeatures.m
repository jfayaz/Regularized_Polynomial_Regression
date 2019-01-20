function [X_poly] = polyFeatures(X, p)
%Maps X (1D vector) into the p-th power
%   [X_poly] = polyFeatures(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% Intialization
X_poly = zeros(numel(X), p);


% Calculating matrix 'X_poly' where the p-th column of X contains the values of X to the p-th power.
for i=1:length(X)
    for j=1:p
        X_poly(i, j) = X(i).^j;
    end
end


end
