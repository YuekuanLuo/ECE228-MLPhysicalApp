
clc
clear all

% making letters
a = [0 1 1 0; 1 0 0 1; 1 1 1 1; 1 0 0 1];
c = [0 1 1 0; 1 0 0 0; 1 0 0 0; 0 1 1 0];
d = [1 1 0 0; 1 0 1 0; 1 0 1 0; 1 1 0 0];
f = [1 1 1 0; 1 0 0 0; 1 1 1 0; 1 0 0 0];
g = [0 1 1 0; 1 0 0 0; 1 0 1 1; 0 1 1 0];
h = [1 0 1 0; 1 0 1 0; 1 1 1 0; 1 0 1 0];
i = [0 0 1 0; 0 0 1 0; 0 0 1 0; 0 0 1 0];
j = [0 0 1 0; 0 0 1 0; 1 0 1 0; 1 1 1 0];
k = [1 0 0 1; 1 0 1 0; 1 1 1 0; 1 0 0 1];
l = [0 1 0 0; 0 1 0 0; 0 1 0 0; 0 1 1 1];
n = [1 0 0 1; 1 1 0 1; 1 0 1 1; 1 0 0 1];
o = [1 1 1 1; 1 0 0 1; 1 0 0 1; 1 1 1 1];
p = [0 1 1 1; 0 1 0 1; 0 1 1 1; 0 1 0 0];
q = [1 1 1 0; 1 0 1 0; 1 1 1 0; 0 0 1 1];
r = [0 1 1 1; 0 1 0 1; 0 1 1 0; 0 1 0 1];
t = [1 1 1 0; 0 1 0 0; 0 1 0 0; 0 1 0 0];
u = [1 0 0 1; 1 0 0 1; 1 0 0 1; 0 1 1 0];
v = [1 0 1 0; 1 0 1 0; 1 0 1 0; 0 1 0 0];
x = [0 0 0 0; 1 0 1 0; 0 1 0 0; 1 0 1 0];
y = [0 0 0 0; 1 0 1 0; 0 1 0 0; 0 1 0 0];

% making 'true' dictionary
Q = [a(:),c(:),d(:),f(:),g(:),h(:),i(:),j(:),k(:),l(:),n(:),o(:),p(:),q(:),r(:),...
    t(:),u(:),v(:),x(:),y(:)];
% Q = Q*diag(1./sqrt(diag(Q'*Q)));


% % visualizing individual letters
% figure(1);
% clf;
% imagesc(y); colormap gray

% figure(2);
% clf;
% imagesc(abs(Q'*Q),[0 1])
% colormap gray

rng(0)
Y = repmat(Q,[1,100]);
noise = .01*randn(size(Y));
Y = Y+noise;

figure(1)
clf;
y = plotDict2D(Y,4);
% h=pcolor(Qim);
imagesc(y,[0 1])
colormap gray
set(gca,'Ytick',[])
set(gca,'Xtick',[])
set(gca,'Ydir','reverse')
% set(h, 'EdgeColor', 'none')
title('original image')

figure(2)
clf;
Qim = plotDict2D(Q,4);
% h=pcolor(Qim);
imagesc(Qim,[0 1])
colormap gray
set(gca,'Ytick',[])
set(gca,'Xtick',[])
set(gca,'Ydir','reverse')
% set(h, 'EdgeColor', 'none')
title('true alphabet')



%% k_svd
% normalize initial, random dictionary
Q0 = randn(16,20);
Q0 = Q0*diag(1./sqrt(diag(Q0'*Q0)));

[Qopt,~]=ksvd02(Y,1,Q0,true)
figure(3)
clf;
Qp = abs(Qopt);
Qp = Qp./max(Qp);


Qim = plotDict2D(abs(Qp),4);
imagesc(Qim)
colormap gray
set(gca,'Ytick',[])
set(gca,'Xtick',[])
title('recovered alphabet')