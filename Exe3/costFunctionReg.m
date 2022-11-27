function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

N = length(y); 
J = 0;
grad = zeros(size(theta));

%phi = 1./(1 + exp(-X*theta));
uni = ones(length(y), 1);
%J = (1/N)*(transpose(y-X*theta)*(y-X*theta)).^2 + lambda*sum(theta.^2);
sig = ones(N,1)./(1+exp(-X*theta));

% Calcolo funzione di costo
b = transpose(y)*log(sig);
c = transpose(ones(N,1)-y)*log(ones(N,1)-sig);
J = -(b+c) + lambda*sum(theta.^2)

%iterations = 5000;
%alpha = 0.001;

%for iter=1:iterations
%    for pp=1:length(theta)
%        %theta(pp) = theta(pp) - alpha*2*((1/N)*(transpose(y-X(:,pp)*theta(pp))*-X(:,pp)) + lambda*theta(pp));
%        theta(pp) = theta(pp) - alpha*(transpose(X(:,pp))*(phi-y)) + 2*lambda*theta(pp);
%        
%    end
%end

%for pp = 1:length(theta)
%    theta(pp) = theta(pp) - alpha*(transpose(X(:,pp)) * (phi-y)) + lambda*2*theta(pp);
%end
grad = transpose(X)*(sig-y) + (lambda*2).*theta;

% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

 

% =============================================================

end
