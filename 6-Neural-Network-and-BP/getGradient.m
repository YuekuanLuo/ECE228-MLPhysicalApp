function [dWi,dWo] = getGradient(Wi,Wo,alpha_i,alpha_o,Inputs,Targets)
%
% Calculate the partial derivatives of the quadratic cost function
% wrt. the weights. Derivatives of quadratic weight decay are included.
%
% Input:
%        Wi      :  Matrix with input-to-hidden weights
%        Wo      :  Matrix with hidden-to-outputs weights
%        alpha_i :  Weight decay parameter for input weights
%        alpha_o :  Weight decay parameter for output weights
%        Inputs  :  Matrix with examples as rows
%        Targets :  Matrix with target values as rows
% Output:
%        dWi     :  Matrix with gradient for input weights
%        dWo     :  Matrix with gradient for output weights

% Determine the number of examples
[exam inp] = size(Inputs);
bias_inp = [Inputs ones(exam,1)];

%%%%%%%%%%%%%%%%%%%%
%%% FORWARD PASS %%%
%%%%%%%%%%%%%%%%%%%%
%
% Propagate examples forward through network
% calculating all hidden- and output unit outputs
% NOTE: tanhf.m has been provided for g(.)
%

% Calculate hidden unit outputs for every example
% Calc h
h_output = bias_inp * Wi';
% Sigmoid
z = tanhf(h_output);
bias_z = [z ones(exam,1)];
  
% Calculate (linear) output unit outputs for every example
% Calc y
y = bias_z * Wo';

%%%%%%%%%%%%%%%%%%%%%
%%% BACKWARD PASS %%%
%%%%%%%%%%%%%%%%%%%%%

% Calculate derivative of 
% by backpropagating the errors from the
% desired outputs

% Output unit deltas
output_delta = y-Targets;
sum_term = bias_z' * output_delta;
dWo = sum_term' + alpha_o * Wo;

% Hidden unit deltas
sum_c = output_delta * Wo;
tmp = (1-bias_z.^2).*sum_c;
dWi = tmp(:,1:end-1)' * bias_inp + alpha_i * Wi;