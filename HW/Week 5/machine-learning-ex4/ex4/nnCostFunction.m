function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.


% Initialize Variables
h = zeros(m,num_labels);
a = cell(1,3);
a{1} = zeros(m,input_layer_size+1);
a{2} = zeros(m,hidden_layer_size+1);
a{3} = zeros(m,num_labels);
z = cell(1,3);
z{1} = zeros(m,input_layer_size);
z{2} = zeros(m,hidden_layer_size);
z{3} = zeros(m,num_labels);
delta = cell(1,3);
delta{3} = zeros(m,num_labels);
delta{2} = zeros(m,hidden_layer_size+1);
Delta = cell(3,1);
Delta{2} = zeros(num_labels,hidden_layer_size+1);
Delta{1} = zeros(hidden_layer_size,input_layer_size+1);

% Expanded labels to format for multiclass classification
y_expanded = zeros(m,num_labels);
for i = 1:m
    y_expanded(i,y(i)) = 1; 
end

% For all samples, propogate through NN
for i = 1:m
    %====================%
    % FORWARD PROPOGATION
    %====================%
    % Forward First Layer
    a{1}(i,:) = [1; X(i,:)'];
    
    % Forward Second Layer
    z{2}(i,:) = a{1}(i,:)*Theta1';
    a{2}(i,:) = [1, sigmoid(z{2}(i,:))];
    
    % Forward Third Layer
    z{3}(i,:) = a{2}(i,:)*Theta2';
    a{3}(i,:) = sigmoid(z{3}(i,:));
    
    % Load into h
    %h(i,:) = a{3}';
    
    %====================%
    % BACK PROPOGATION
    %====================%
    
    % Backward third layer
    delta{3}(i,:) = a{3}(i,:)-y_expanded(i,:);
    
    % Backward second layer
    delta{2}(i,:) = Theta2'*delta{3}(i,:)'.*sigmoidGradient([1,z{2}(i,:)]');
    
    % Accumulate the gradient
    Delta{2} = Delta{2} + delta{3}(i,:)'*a{2}(i,:);
    Delta{1} = Delta{1} + delta{2}(i,2:end)'*a{1}(i,:);
end

h = a{3};

% Compute Total Cost
tmp2 = -y_expanded.*log(h) - (1-y_expanded).*log(1-h);
J_unreg = 1/m*(sum(sum(tmp2)));

% Compute cost of regularization
reg_cost = lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
J = J_unreg + reg_cost;

% Obtain unregularized gradient for neural network cost function
Delta{2} = 1/m * Delta{2};
Delta{1} = 1/m * Delta{1};

Theta1_grad = Delta{1}(:,1:end) + lambda / m * Theta1(:,1:end);
Theta2_grad = Delta{2}(:,1:end) + lambda / m * Theta2(:,1:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
