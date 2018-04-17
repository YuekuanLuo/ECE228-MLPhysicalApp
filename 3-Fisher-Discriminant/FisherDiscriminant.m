%%%%%%%%%%%%%%%%%%%%%%% Parameters (Don't change)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  mu1=[5 4]';	    % true mean value
  mu2=[4 -1]';	    % true mean value

  SIGMA=[2   -1.7 
    -1.7 3  ];   % true covariance matrix

  p1 = 0.3;          % P(C1)  probabilty of class 1
  p2 = 1-p1;

  N=4000;            % number of points in density

  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

d1 = randmvn(mu1,SIGMA,N);  % A random sample
d2 = randmvn(mu2,SIGMA,N);  % A random sample

% randbin(p1,1,N) is creating a 1xN vector where each entry has a probability p1 of being 1 probability 1-p1 of being 0
c1 = randbin(p1,1,N);        % Occurrence of class 1
c2 = 1 - c1;                 % Occurrence of class 2

X  = d1.*[c1;c1] + d2.*[c2; c2];

% this is where your code goes (generate the appropriate weight vector "w")
c1_data = [X(1,logical(c1));
           X(2,logical(c1))];
c2_data = [X(1,logical(c2));
           X(2,logical(c2))];

ave_1 = mean(c1_data');
ave_2 = mean(c2_data');

S = cov(c1_data') + cov(c2_data');

w = inv(S)*(ave_2-ave_1)';

Y = w'*X;

%%%%%%%%%%%%%% Eigenvector Transformation %%%%%%%%%%

% copy & paste your code from the PCA for coordinate transform assignment here (need solution for X_)

m = numel(X);
ave = mean(mean(X));
cov_matrix = (1/m)*(X-ave)*transpose((X-ave));
[U,D] = eig(cov_matrix);
lambda = trace(D);
X_ =  U * X; % Coordinate transformed set

fprintf('Calculating histograms ... ');
  nbins = 50;                   % # bins in histogram 
  [n1_ x1_] = hist(X_(1,:),nbins); % Marginal Histograms
  [n2_ x2_] = hist(X_(2,:),nbins);
  n1_ = n1_/sum(n1_);  
  n2_ = n2_/sum(n2_);  
  [n_y y1_] = hist(Y,nbins);  % histogram of Fisher transformed 
  n_y = n_y/sum(n_y);  
fprintf('Done\n\r');

%%%%%%%%%%%%%%%%%%%%%% Results %%%%%%%%%%%%%%%%%%%
disp('The calculated weight-vector is ');
w
disp('The eigenvectors and eigenvalues are ');
U
lambda
N

%%%%%%%%%%%%%%%%%% Plotting 3D %%%%%%%%%%%%%%%%%%%

figure(1)
clf
subplot 221
  plot(X(1,logical(c1)),X(2,logical(c1)),'b.');     %Class 1
  hold on
  plot(X(1,logical(c2)),X(2,logical(c2)),'r.');     %Class 2
  ax=axis;
  axis equal;
  axis(ax);
  h = get(gca,'position');
  w1=w*0.3*diff(xlim);  
  xlabel('x_1')
  ylabel('x_2')  
  title('Scatter plot of data-set');
  temp(1) = 0.01;
subplot 222
  bar(y1_, n_y,1,'w');
  xlabel('y');
  ylabel('p(y)');
  title('Marginal Distribution after Fisher transf.');  
subplot 223
  bar(x1_,n1_,1,'w');
  ylim([0 max(n1_)*1.02]);    
  xlabel('x_1')
  ylabel('p(x_1)')
  title('Marginal Distribution after Eigenv. transf.');    
subplot 224
  bar(x2_,n2_,1,'w');
  ylim([0 max(n2_)*1.02]);    
  xlabel('x_2')  
  ylabel('p(x_2)')
  title('Marginal Distribution after Eigenv. transf.');