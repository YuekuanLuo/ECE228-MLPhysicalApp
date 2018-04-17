rng(0)                 %set random seed
%%%%%%%%%%%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mu1=[0 0]';	    % true mean value
mu2=[5 5]';	    % true mean value

SIGMA=[2 0 ; 0 2];   % true covariance matrix

p1 = 0.3;          % P(C1)  probabilty of class 1
p2 = 1-p1;

N=4000;            % number of points in density
  
X = GenSet(mu1,mu2,SIGMA,N,p1); % generate dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% your code goes here, solve for the transformed set X_
% by multiplying X by the eigenvectors of its sample covariance
% matrix. Hint: Don't forget to subtract the mean of X
m = numel(X);
ave = mean(mean(X));
cov_matrix = (1/m)*(X-ave)*transpose((X-ave));
[U,D] = eig(cov_matrix);
lambda = trace(D);
X_ =  U * X; % Coordinate transformed set
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Calculating histograms ... ');
  nbins = 50;                   % # bins in histogram 
  [n1 x1] = hist(X(1,:),nbins); % Marginal Histograms
  [n2 x2] = hist(X(2,:),nbins);
  [n1_ x1_] = hist(X_(1,:),nbins);
  [n2_ x2_] = hist(X_(2,:),nbins);
  n1 = n1/sum(n1);  
  n2 = n2/sum(n2);  
  n1_ = n1_/sum(n1_);  
  n2_ = n2_/sum(n2_);  
fprintf('Done\n\r');
%%%%%%%%%%%%%%%%%%%%%% Results %%%%%%%%%%%%%%%%%%%
disp('The eigenvectors and eigenvalues are ');
U
lambda
%%%%%%%%%%%%%%%%%%% Plotting 3D %%%%%%%%%%%%%%%%%%%

figure(1)
clf
subplot 221
  bar(x1,n1,1,'w');
  ylim([0 max(n1)*1.02]);    
  xlabel('x_1')
  ylabel('p(x_1)')
  title('Marginal Distr. before Eigenv. transf.');    
subplot 222
  bar(x2,n2,1,'w');
  ylim([0 max(n2)*1.02]);    
  xlabel('x_2')  
  ylabel('p(x_2)')
  title('Marginal Distr. before Eigenv. transf.');   
subplot 223
  bar(x1_,n1_,1,'w');
  ylim([0 max(n1_)*1.02]);    
  xlabel('x_1')
  ylabel('p(x_1)')
  title('Marginal Distr. after Eigenv. transf.');    
subplot 224
  bar(x2_,n2_,1,'w');
  ylim([0 max(n2_)*1.02]);    
  xlabel('x_2')  
  ylabel('p(x_2)')
  title('Marginal Distr. after Eigenv. transf.');   