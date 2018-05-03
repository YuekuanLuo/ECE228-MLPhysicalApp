%  MATLAB main program for neural network training
%
%  The network is trained using gradient method.
%  Sunspot series is used as dataset.

clear
d=5;                     % Sunspot input window 
Ni = d;                  % Number of external inputs
Nh = 8;                  % Number of hidden units
No = 1;                  % Number of output units
alpha_i = 0.0;           % Input weight decay
alpha_o = 0.0;           % Output weight decay
eta = 0.005;             % Learnstep size
greps = 1e-3;            % Gradient norm stopping criteria
range = 0.1;             % Initial weight range                
max_iter=500;           % maximum number of iterations

rng('default');


% First, get some data...
[train_inp,train_tar,test_inp,test_tar] = getsun(d);
ptrain = length(train_inp);             % Number of training examples
ptest = length(test_inp);               % Number of test examples

% compute the signal variance for normalization;
sigvar=train_tar-ones(ptrain,1)*mean(train_tar);
sigvar=sum(sum(sigvar.*sigvar))/ptrain;
errnorm=2/sigvar;

% Initialize network weights
Wi = range * randn(Nh,Ni+1);
Wo = range * randn(No,Nh+1);

%% Student code goes here -> Implement gradient descent for max_iterations to find the optimal Wi and Wo

bias_inp = [train_inp ones(ptrain,1)];

iter = 0;
saved_err = 0;
diff = 0;
while diff > greps || iter < 2
    iter = iter + 1;
    [dWi,dWo] =  getGradient(Wi,Wo,alpha_i,alpha_o,train_inp,train_tar);
    Wi = Wi - eta .* dWi;
    Wo = Wo - eta .* dWo;
    z = bias_inp * Wi';
    bias_z = [z ones(ptrain,1)];
    output = bias_z * Wo';
    err = sum((output - train_tar).^2)/2 + 0.5 * sum(sum(Wi.^2)) + 0.5 * sum(sum(Wo.^2));
    diff = saved_err - err;
    saved_err = err;
end
