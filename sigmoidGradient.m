function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. Z is a matrix or vector. 

g = zeros(size(z));
gFun = @(x) 1./(1 + exp(-x));
g = gFun(z).*(1-gFun(z));

end
