clear all; 
clc
close all

addpath 'nmflib/'

%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%

%% Load Data
load('Datasets/MNSIT_2digits_100.mat');
load('Datasets/Exact_20bases_MNSIT_2digits_100.mat');
X=X';

%% Create noisy data
A=[];
sigma=0.25;
for l=1:10
    X1=X+sigma*abs(randn(size(X)));  %--Absolute Gaussian noise
   %X1=X+mat2gray(poissrnd(sigma,size(X))); %--Poisson Noise
    A=[A;X1];
end
%% UoI_NMF
%--Parameters
params.k=20;       % Rank k of the factorization
params.B1 =20;     % number of bootstrap samples for selection
params.B2 =10;     % number of bootstrap samples for bagging (estimation)

params.epsilon=0.3; % Density parameter in DBSCAN
params.MinPts =  params.B1/4; %Minimum points in a cluster
  
[output1] =  UoINMF_KL_DBcluster(A,params);
W1=output1.W;
H1=output1.H;
k=size(H1,1);

%% Evaluation
%--- calculate nonzeros and errors
nnzW1=median(sum(abs(W1)>1.0,2));
nnzH1=median(sum(abs(H1)>0.05,2));

Aest=W1*H1;
%%--plot basis and bases quality
figure()
for i=1:16
    subplot(5,4,i)
     H11(i,:)=mat2gray(H1(i,:));
     imshow((reshape(H11(i,:),[],56)));
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
legend('UoI-NMF')
xlabel('Bases')
ylabel('Correlation with exact bases')
title('Correlation between exact and learned bases')
%
%
figure()
plot(sort(rms1),'r*-')
legend('UoI-NMF')
xlabel('Bases')
ylabel('Mean Squared Error')
title('Mean Squared Error between exact and learned bases')

%---- Error calculation

eer1=norm(A-Aest,'fro') %% reconstruction error with noisy data


for l=1:size(X)
    Wnew1(l,:)= lsqnonneg(H1',X(l,:)')';
end

er1=norm(X-Wnew1*H1,'fro')  %% reconstruction error with original data

%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%
%-%-%-%-%-%-%-%-%-%-%-%        Daming         %-%-%-%-%-%-%-%-%-%-%-%-%-%
%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%-%

param.K=20;  % learns a dictionary with 100 elements
param.lambda=0.15;
param.numThreads=-1; % number of threads
param.batchsize=5;
param.verbose=false;

param.iter=1000;  % let us see what happens after 1000 iterations.

% let us add sparsity to the dictionary itself
fprintf('*********** THIRD EXPERIMENT ***********\n');
param.modeParam=0;
param.iter=1000;
param.gamma1=2.7;
param.modeD=1;

tic

% [D] = mexTrainDL(X,param);
[D] = mexTrainDL(A,param);
t=toc;
fprintf('time of computation for Dictionary Learning: %f\n',t);
fprintf('Evaluating cost function...\n');
alpha=mexLasso(X,D,param);
R=mean(0.5*sum((X-D*alpha).^2)+param.lambda*sum(abs(alpha)));
fprintf('objective function: %f\n',R);
tic

%% Show first 20 2-digit images
figure('Name','2-D MNIST dictionary atoms')
for i=1:20
%     fprintf('%d. Hello world!\n', i);
    subplot(4,5,i)
    row1matfull = full(reshape(alpha(i,:),[],56));
    imshow(row1matfull);
end

nnzW1=median(sum(abs(D)>1.0,2));
nnzH1=median(sum(abs(alpha)>0.05,2));
Aest=D*alpha;
alpha = full(alpha);
%%--plot basis and bases quality
figure()
for i=1:16
    subplot(5,4,i)
     alpha1(i,:)=mat2gray(alpha(i,:));
     imshow((reshape(alpha1(i,:),[],56)));
end
%%-- Correlation with the exact bases
C1=normc(alpha*Hopt');
[corr1,id1]=max(C1,[],2);

Hl1=[];j1=[];
for j=1:k
    i1=find(id1==j);
    if numel(i1)==0
        j1=[j1;j];
        continue;
    elseif numel(i1)==1
        rms1(j,1)=mean((alpha(i1,:)-Hopt(j,:)).^2,2);
    else
        [~,i2]=max(corr1(i1));
        rms1(j,1)=mean((alpha(i1(i2),:)-Hopt(j,:)).^2,2);
        Hl1=[Hl1;alpha(setdiff(i1,i1(i2)),:)];
    end
end


figure()
plot(sort(abs(corr1)),'r*-')
legend('ODL')
xlabel('Bases')
ylabel('Correlation with exact bases')
title('Correlation between exact and learned bases')
%
%
figure()
plot(sort(rms1),'r*-')
legend('ODL')
xlabel('Bases')
ylabel('Mean Squared Error')
title('Mean Squared Error between exact and learned bases')

%---- Error calculation

eer1=norm(A-Aest,'fro') %% reconstruction error with noisy data


for l=1:size(X)
    Wnew1(l,:)= lsqnonneg(H1',X(l,:)')';
end

er1=norm(X-Wnew1*H1,'fro')  %% reconstruction error with original data
return;
