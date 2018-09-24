clc
close all
clear all

addpath 'nmflib/'

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

%[mdlstrct] = basicNMF_KL_UoI_corrthres_new(A,maxk,0.92);
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

 eer1=norm(A-Aest,'fro'); %% reconstruction error with noisy data

 
  for l=1:size(X)
            Wnew1(l,:)= lsqnonneg(H1',X(l,:)')';
  end

er1=norm(X-Wnew1*H1,'fro');  %% reconstruction error with original data


%% Visualization
% [n,d]=size(A);
% r=randperm(n);
% % W1=mdlstrct.opt.Wopt;
% figure()
% l1=0;
% for l=1:5
%     for i=1:4
%      subplot(5,16,i+l1)
%     imshow((reshape(H1(i+l1/4,:),[],56)));
%      subplot(5,16,i+l1+4)
%     image(imcomplement(mat2gray(reshape((abs(W1(r(i+l1/4),:))>5.01),4,[])))','CDataMapping','scaled');
%     subplot(5,16,i+l1+8)
%     imshow(mat2gray(reshape(A(r(i+l1/4),:),[],56)));
%      subplot(5,16,i+l1+12)
%     imshow((reshape(A(r(i+l1/4),:),[],56)));
%     end
%     l1=l1+16;
% end
% set(gca,'LooseInset',get(gca,'TightInset'));

