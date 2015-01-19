function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
  %theta = theta - (rand(size(theta)) - 0.5) * 0.1;
  theta_norm = sum(exp(theta' * X));
  theta_prod = exp(theta' * X);
  P = bsxfun(@rdivide,theta_prod, theta_norm);
  P_log = [log(P); zeros(1,m)];
  I=sub2ind(size(P_log), y, 1:size(P_log,2));
  values = P_log(I);
  indexes = zeros(size(P_log));
  indexes(I) = 1;
  dev_sub = bsxfun(@minus,indexes(1:end-1, :), P);

  f = -sum(values);
  g = - X * dev_sub';
  g=g(:); % make gradient a vector for minFunc

