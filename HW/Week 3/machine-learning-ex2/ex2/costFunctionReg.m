function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% J
tmp1 = -y.*log(sigmoid(X*theta)) - ((1-y).*log(1-sigmoid(X*theta)));
tmp2 = theta.^2;
J = 1/m * sum(tmp1(1:end)) + lambda/(2*m) * sum(tmp2(2:end));

% grad(1)
grad(1) = 1/m*sum(sigmoid(X*theta)-y);

% grad(2)
n = size(X,2);
tmp1 = repmat(sigmoid(X*theta)-y,1,n).*X;
tmp2 = (1/m * sum(tmp1,1))' + lambda/m*theta;
grad(2:end) = tmp2(2:end);
% =============================================================

end
