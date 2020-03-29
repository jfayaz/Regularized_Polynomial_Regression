clear ; close all; clc; addpath('functions')
%% =========== Polynomial Regression ========= %%
%  written by : JAWAD FAYAZ (email: jfayaz@uci.edu)
%  visit: (https://jfayaz.github.io)

%  ------------- Instructions -------------- %
%  INPUT:
%  Input Variables should include:
%  "Exdata"  --> must be in the form of .mat file and must be in same directory and should include following variables:
%        'X'      -->  (m,1) vector containing Train data 
%        'y'      -->  (m,1) vector containing Train data
%        'Xval'   -->  (n,1) vector containing Cross-Validation data
%        'yval'   -->  (n,1) vector containing Cross-Validation data
%        'Xtest'  -->  (n,1) vector containing Test data
%        'ytest'  -->  (n,1) vector containing Test data
%  "Order_of_Polynomial"  --> Value of order of the polynomial that user wishes to fit
%  "lambda"               --> Regularized Regression parameter (if user is not sure what to use, the last part of this code 'Validation for Selecting Lambda' 
%                             will help in selecting best value of lambda- Figure 3 will display the variation of Error w.r.t lambda)
%
%  OUTPUT:
%  Output will be provided in following variables:  
%  "theta"        --> Vector (Order_of_Polynomial+1,1) containing the Polynomial Regression coeffecients
%  "mu"           --> Vector (Order_of_Polynomial,1)
%  "sigma"        --> Vector (Order_of_Polynomial,1)
%  "lambda_vec"   --> Vector (10,1) testing different Regularized regression parameter (Figure 3)

%%%%% ============================================================= %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ====================== USER INPUTS =============================== %%

%%% Provide your .mat file name here  
Matlab_Data_Filename = 'Exdata.mat';

%%% Order of the Polynomial to fit
Order_of_Polynomial = 8;

%%% Regularized regression parameter 
%%% (if you dont know what to use, the last part of this code 'Validation for Selecting Lambda' 
%%% will help you selecting best value of lambda- Figure 3 will display the variation of Error w.r.t lambda)
lambda = 0;


%%%%%%================= END OF USER INPUT ========================%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ---------- Plotting Data ----------
load (Matlab_Data_Filename);

% m = Number of examples
m = size(X, 1);

% Plot training data
figure(1)
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('X','fontWeight','bold')
ylabel('y','fontWeight','bold')
set(gca,'fontsize',14,'FontName', 'Times New Roman','LineWidth', 1.25,'TickDir','out','TickLength', [0.005 0.005])
grid on; box on;

%% ---------- Feature Mapping for Polynomial Regression ------
p = Order_of_Polynomial;

% Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(m, 1), X_poly];                   % Add Ones

% Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p);
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones

% Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones

fprintf('Normalized Training Example 1:\n');
fprintf('  %f  \n', X_poly(1, :));


%% ---------- Learning Curve for Polynomial Regression ------
[theta] = trainLinearReg(X_poly, y, lambda);

% Plot training data and fit
plotFit(min(X), max(X), mu, sigma, theta, p);
title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));

figure(2);
[error_train, error_val] = learningCurve(X_poly, y, X_poly_val, yval, lambda);
plot(1:m, error_train, 1:m, error_val, 'LineWidth', 1.5);

title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of Training Examples','fontWeight','bold')
ylabel('Error','fontWeight','bold')
legend('Train', 'Cross Validation')
set(gca,'fontsize',14,'FontName', 'Times New Roman','LineWidth', 1.25,'TickDir','out','TickLength', [0.005 0.005])
grid on; box on;

fprintf('Polynomial Regression (lambda = %f)\n\n', lambda);
fprintf('# Training_Examples\tTrain_Error\tCross-Validation_Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end


%% ---------- Validation for Selecting Lambda ------
%  Implementing validationCurve to test various values of lambda on a validation set. 
%  You can then use this to select the "best" lambda value.
%

[lambda_vec, error_train, error_val] = validationCurve(X_poly, y, X_poly_val, yval);

figure(3)
plot(lambda_vec, error_train, lambda_vec, error_val,'LineWidth', 1.5);
legend('Train', 'Cross Validation');
xlabel('lambda','fontWeight','bold');
ylabel('Error','fontWeight','bold');
set(gca,'fontsize',14,'FontName', 'Times New Roman','LineWidth', 1.25,'TickDir','out','TickLength', [0.005 0.005])
grid on; box on;

fprintf('lambda\t\tTrain_Error\tValidation_Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n',lambda_vec(i), error_train(i), error_val(i));
end

