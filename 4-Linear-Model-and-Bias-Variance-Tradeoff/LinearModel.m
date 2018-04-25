warning off
clear all
close all
rng('default')   %fix random generator seed

%%%%%%%%%%%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w               = [1 2 0.5]';	 % True weights
noiselevel      = 0.75;          % Standard deviation of Gaussian noise on data
d               = length(w);     % Number of dimensions
Nmin            = 1;             % Minimal training set size
Nmax            = 200;            % Maximal training set size
N               = 10000;         % Samples in dataset
repetitions     = 10;            % Number of repetitions

%% Simulate linear dataset
for j = 1:repetitions   
    X1              = [ones(N,1),randn(N,d-1)]';          % model 1: all inputs X known
    X2              = X1(1:d-1,:);                        % model 2: one input unknown
    noiselevel      = .75;                                % arbitrary noise strength
    n               = noiselevel*randn(N,1);              % noise
    y               = X1'*w + n;                          % simulated measurements
    c               = 1;                                  % counter
    
    for i = Nmin:Nmax
        Ntrain          = i;                           % number of measurements to be used
        Ntest           = N-Ntrain;                       % Test set size
        X1_train        = X1(:,1:Ntrain);                 % Training set, X1
        X2_train        = X2(:,1:Ntrain);                 % Training set, X2
        
        %% Student code, compute optimal weight estimates
        
        w1              = inv(X1_train * X1_train') * X1_train * y(1:Ntrain);    %Least squares estimated weights, model 1
        w2              = inv(X2_train * X2_train') * X2_train * y(1:Ntrain);    %Least squares estimated weights, model 2
        
        y1_est          = w1'*X1(:,Ntrain+1:end);   % Estimated measurements given model 1
        y2_est          = w2'*X2(:,Ntrain+1:end);    % Estimated measurements given model 2
        y_test          = y(Ntrain+1:end)';    % Test set
        y_train         = y(1:Ntrain);    % Train set
        
        %% end student code
        
        err1_test       = mean((y_test-y1_est).^2);            % Test error model 1
        err2_test       = mean((y_test-y2_est).^2);            % Test error model 2
        err1_train      = mean((y_train-X1(:,1:Ntrain)'*w1).^2);% Training error model 1
        err2_train      = mean((y_train-X2(:,1:Ntrain)'*w2).^2);% Training error model 2
        
        test1(j,c)      = err1_test;
        test2(j,c)      = err2_test;
        train1(j,c)     = err1_train;
        train2(j,c)     = err2_train;
        Ns(c)           = Ntrain;
        c               = c+1;
    end
    
    
end

figure(1), 
hold off,
h1=errorbar(Ns,mean(train1),std(train1)/sqrt(repetitions),'r:');
hold on
h2=errorbar(Ns,mean(train2),std(train2)/sqrt(repetitions),'b:');
%h3=errorbar(Ns,mean(test1),std(test1)/sqrt(repetitions),'r');
% hold on
%h4=errorbar(Ns,mean(test2),std(test2)/sqrt(repetitions),'b');
legend('train 1', 'train 2')%,'test 1', 'test 2')
xlabel('training set size')
ylabel('mean square errors (training)')
grid on

figure(2), 
hold off,
% h1=errorbar(Ns,mean(train1),std(train1)/sqrt(repetitions),'r:');
% hold on
% h2=errorbar(Ns,mean(train2),std(train2)/sqrt(repetitions),'b:');
h3=errorbar(Ns,mean(test1),std(test1)/sqrt(repetitions),'r');
hold on
h4=errorbar(Ns,mean(test2),std(test2)/sqrt(repetitions),'b');
legend('test 1', 'test 2')
xlabel('training set size')
ylabel('mean square errors (testing)')
grid on
