% clear all; 
% clc
% close all



%% Load Data
load('Datasets/Swimmer.mat');
load('Datasets/Exact_16bases_swimmer.mat');
X=reshape(mat2gray(Y),[],256,1)';

%% Create noisy data
A=[];
sigma=0.25; %% noise level
for l=1:10
    X1=X+sigma*abs(randn(size(X)));  %--Absolute Gaussian noise
   %X1=X+mat2gray(poissrnd(sigma,size(X))); %--Poisson Noise
    A=[A;X1];
end

X_orig = X;
X = A;

%% ODL

param.K=16;  % learns a dictionary with 100 elements
param.lambda=0.15;
param.lambda=0.04;
param.lambda=0.02;

param.numThreads=-1; % number of threads
param.batchsize=10;
param.verbose=false;

param.iter=1000;  % let us see what happens after 1000 iterations.

% let us add sparsity to the dictionary itself
fprintf('*********** THIRD EXPERIMENT ***********\n');
param.modeParam=0;
param.iter=1000;
param.gamma1=3;
param.modeD=1;

tic

[D] = mexTrainDL(X,param);
t=toc;
fprintf('time of computation for Dictionary Learning: %f\n',t);
fprintf('Evaluating cost function...\n');
alpha=mexLasso(X,D,param);
R=mean(0.5*sum((X-D*alpha).^2)+param.lambda*sum(abs(alpha)));
fprintf('objective function: %f\n',R);
tic

% Aest=D*alpha;
H1 = full(alpha);
W1 = D;
Aest=W1*H1;
k=size(H1,1);

%%--plot basis and bases quality
figure()
for i=1:k
    subplot(4,4,i)
    H11(i,:)=mat2gray(H1(i,:));
    imshow((reshape(H11(i,:),[],32)));
    %colorbar; colormap(noeljet)
    %colormap('default')
end

%%--plot basis and bases quality
figure('Name','2-D MNIST dictionary atoms')
title('ODL basis and bases quality')

for i=1:k
    subplot(4,4,i)
    H11(i,:)=mat2gray(H1(i,:));
    imshow((reshape(H11(i,:),[],32)));
end

%%-- Correlation with the exact bases
C1=normc(H11*Hopt');
[corr1,id1]=max(C1,[],2);

Hl1=[];j1=[];
for j=1:k
    i1=find(id1==j);
    if numel(i1)==0
        j1=[j1;j];
        continue;
    elseif numel(i1)==1
        rms1(j,1)=mean((H11(i1,:)-Hopt(j,:)).^2,2);
    else
        [~,i2]=max(corr1(i1));
        rms1(j,1)=mean((H11(i1(i2),:)-Hopt(j,:)).^2,2);
        Hl1=[Hl1;H11(setdiff(i1,i1(i2)),:)];
    end
end


figure()
plot(sort(abs(corr1)),'r*-')
% legend('Online Dictionary Learning')
hold on

plot(sort(abs(corr1_swimmer_uoi)),'bo-')
legend('Online Dictionary Learning', 'UoI-NMF','Location','southeast')

xlabel('Bases')
ylabel('Correlation with exact bases')
title('Correlation between exact and learned bases')
hold off


figure()
plot(sort(rms1),'r*-')
% legend('Online Dictionary Learning')

hold on

plot(sort(rms1_swimmer_uoi),'bo-')
legend('Online Dictionary Learning', 'UoI-NMF','Location','southeast')

xlabel('Bases')
ylabel('Mean Squared Error')
title('Mean Squared Error between exact and learned bases')
hold off

% ---- Error calculation

eer1=norm(A-Aest,'fro') %% reconstruction error with noisy data

for l=1:size(X)
    Wnew1(l,:)= lsqnonneg(H1',X(l,:)')';
end

er1=norm(X-Wnew1*H1,'fro')  %% reconstruction error with original data

aaaaa = 12*22;
aaaaa = 12*22;
aaaaa = 12*22;

return;
