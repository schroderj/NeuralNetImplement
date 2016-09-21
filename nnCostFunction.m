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

%change y to be binary matrix symbolizing the resulting value
yMatrix = zeros(length(y),num_labels);
for i = 1:length(y)
    index = y(i);
    yMatrix(i, index) = 1;
end

%add the initial value rows to the X vector
dummy = ones(size(X, 1), 1);
X = [dummy, X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

%% cost function
%unregularized is calculated here
for i = 1:m
    
    a2 = zeros(1,size(Theta1, 1));
    for j = 1:size(Theta1, 1)
        a2(j) = Theta1(j,:)*X(i,:)';
    end

    a2 = sigmoid(a2);
    a2 = [1, a2];
    
    a3 = zeros(num_labels,1);
    for j = 1:size(Theta2, 1)
        a3(j) = Theta2(j,:)*a2';
    end
    
    a3 = sigmoid(a3)';
    
    observationSum = 0;
    for k = 1:num_labels
        costTerm = -1 * yMatrix(i,k) * log(a3(k)) - (1 - yMatrix(i,k)) * log(1-a3(k));
        observationSum = observationSum + costTerm;
    end
    
    J = J + observationSum;

end
J = J/m;

%calculate the regularized term
sumThetas1 = 0;
for j = 1:hidden_layer_size
    for k = 2:(input_layer_size+1)
        theta = Theta1(j,k)^2;
        sumThetas1 = sumThetas1 + theta;
    end
end

sumThetas2 = 0;
for j = 1:num_labels
    for k = 2:(hidden_layer_size+1)
        theta = Theta2(j,k)^2;
        sumThetas2 = sumThetas2 + theta;
    end
end

regularizedTerm = (sumThetas1 + sumThetas2)*lambda/(2*m);

J = J + regularizedTerm;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

DELTA1 = zeros(size(Theta1_grad));
DELTA2 = zeros(size(Theta2_grad));
for t = 1:m
    
    %step 1
    a1 = X(t,:);
    z2 = zeros(1,size(Theta1, 1));
    for j = 1:size(Theta1, 1)
        z2(j) = Theta1(j,:)*a1';
    end
    a2 = sigmoid(z2);
    a2 = [1, a2];
    
    z3 = zeros(num_labels, 1);
    for j = 1:size(Theta2, 1)
        z3(j) = Theta2(j,:)*a2';
    end
    a3 = sigmoid(z3)';

    %step 2
    deltak3 = a3-yMatrix(t,:);

    delta2 = (Theta2'*deltak3');
    delta2 = delta2(2:end);
    delta2 = delta2.*sigmoidGradient(z2)';
    
    %step 4
    DELTA1 = DELTA1 + delta2*a1;
    DELTA2 = DELTA2 + deltak3'*a2;
 
end


D1 = DELTA1./m;
D2 = DELTA2./m;

regularizedD1 = lambda/m.*Theta1;
regularizedD1(:,1) = zeros(size(regularizedD1,1),1);
regularizedD2 = lambda/m.*Theta2;
regularizedD2(:,1) = zeros(size(regularizedD2,1),1);

D1 = D1 + regularizedD1;
D2 = D2 + regularizedD2;


Theta1_grad = D1;
Theta2_grad = D2;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
