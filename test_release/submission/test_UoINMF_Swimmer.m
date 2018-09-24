clc
close all
clear all

addpath 'nmflib/'

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
%% UoI_NMF
%--Parameters
 params.k=16;       % Rank k of the factorization
 params.B1 =20;     % number of bootstrap samples for selection
 params.B2 =10;     % number of bootstrap samples for bagging (estimation)

params.epsilon=0.3; % Density parameter in DBSCAN
params.MinPts =  params.B1/4; %Minimum points in a cluaster
  
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
for i=1:k
    subplot(4,4,i)
    H11(i,:)=mat2gray(H1(i,:));
    imshow((reshape(H11(i,:),[],32)));
    %colorbar; colormap(noeljet)
    %colormap('default')
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
% r=randperm(n);
% W1=mdlstrct.opt.Wopt;
% figure()
% l1=0;
% for l=1:4
%     for i=1:4
%      subplot(4,16,i+l1)
%     imshow((reshape(H11(i+l1/4,:),[],32)));
%      subplot(4,16,i+l1+4)
%     image(imcomplement(mat2gray(reshape((abs(W1(r(i+l1/4),:))>2.01),4,[])))','CDataMapping','scaled');
%     subplot(4,16,i+l1+8)
%     imshow(mat2gray(reshape(A11(r(i+l1/4),:),[],32)));
%      subplot(4,16,i+l1+12)
%     imshow((reshape(A(r(i+l1/4),:),[],32)));
%     end
%     l1=l1+16;
% end
% set(gca,'LooseInset',get(gca,'TightInset')); 
%  for i=1:16
%      Dis1(:,:,1,i)=reshape(H11(i,:),[],32);
%      Dis2(:,:,1,i)=imcomplement(mat2gray(reshape((abs(W1(r(i),:))>1.01),4,[])));
%      Dis3(:,:,1,i)=mat2gray(reshape(A11(r(i),:),[],32));
%      Dis4(:,:,1,i)=mat2gray(reshape(A(r(i),:),[],32));
%  end
%   subplot(1,4,1)
% montage(Dis1, 'Size', [4 4]);
% subplot(1,4,2)
% montage(Dis2, 'Size', [4 4]);
% subplot(1,4,3)
% montage(Dis3, 'Size', [4 4]);
% subplot(1,4,4)
% montage(Dis4, 'Size', [4 4]);