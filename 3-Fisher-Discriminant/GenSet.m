function X = GenSet(mu1,mu2,SIGMA,N,p1)
d1 = randmvn(mu1,SIGMA,N);  % A random sample
d2 = randmvn(mu2,SIGMA,N);  % A random sample

c1 = randbin(p1,1,N);        % Occurrence of class 1
c2 = 1 - c1;                 % Occurrence of class 2

X  = d1.*[c1;c1] + d2.*[c2; c2];