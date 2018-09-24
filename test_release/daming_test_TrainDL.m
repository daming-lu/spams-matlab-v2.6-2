clear all; 

% I=double(imread('data/lena.png'))/255;
I=double(imread('lena.png'))/255;

% extract 8 x 8 patches
X=im2col(I,[8 8],'sliding');
X=X-repmat(mean(X),[size(X,1) 1]);
X=X ./ repmat(sqrt(sum(X.^2)),[size(X,1) 1]);


%% Load Data
load('Datasets/MNSIT_2digits_100.mat');
load('Datasets/Exact_20bases_MNSIT_2digits_100.mat');
X=X';

param.K=20;  % learns a dictionary with 100 elements
param.lambda=0.15;
param.numThreads=-1; % number of threads
param.batchsize=5;
param.verbose=false;

param.iter=1000;  % let us see what happens after 1000 iterations.

%%%%%%%%%% FIRST EXPERIMENT %%%%%%%%%%%
% tic
% D = mexTrainDL(X,param);
% t=toc;
% fprintf('time of computation for Dictionary Learning: %f\n',t);
% 
% fprintf('Evaluating cost function...\n');
% alpha=mexLasso(X,D,param);
% R=mean(0.5*sum((X-D*alpha).^2)+param.lambda*sum(abs(alpha)));
% % ImD=displayPatches(D);
% ImD = D;
% % subplot(1,3,1);
% 
% %% Show first 20 2-digit images
% figure('Name','1st exp')
% for i=1:20
% %     fprintf('%d. Hello world!\n', i);
%     subplot(4,5,i)
%     row1matfull = full(reshape(alpha(i,:),[],56));
%     imshow(row1matfull);
% end
% 
% % imagesc(ImD); colormap('gray');
% % fprintf('objective function: %f\n',R);
% % drawnow;
% 
% fprintf('*********** SECOND EXPERIMENT ***********\n');
% %%%%%%%%%% SECOND EXPERIMENT %%%%%%%%%%%
% % Train on half of the training set, then retrain on the second part
% X1=X(:,1:floor(size(X,2)/2));
% X2=X(:,floor(size(X,2)/2):end);
% param.iter=500;
% tic
% [D model] = mexTrainDL(X1,param);
% t=toc;
% fprintf('time of computation for Dictionary Learning: %f\n',t);
% fprintf('Evaluating cost function...\n');
% alpha=mexLasso(X,D,param);
% 
% % R=mean(0.5*sum((X-D*alpha).^2)+param.lambda*sum(abs(alpha)));
% % fprintf('objective function: %f\n',R);
% tic
% % Then reuse the learned model to retrain a few iterations more.
% param2=param;
% param2.D=D;
% [D model] = mexTrainDL(X2,param2,model);
% %[D] = mexTrainDL(X,param);
% t=toc;
% fprintf('time of computation for Dictionary Learning: %f\n',t);
% fprintf('Evaluating cost function...\n');
% alpha=mexLasso(X,D,param);
% R=mean(0.5*sum((X-D*alpha).^2)+param.lambda*sum(abs(alpha)));
% fprintf('objective function: %f\n',R);
% 
% %% Show first 20 2-digit images
% figure('Name','2nd exp')
% for i=1:20
% %     fprintf('%d. Hello world!\n', i);
%     subplot(4,5,i)
%     row1matfull = full(reshape(alpha(i,:),[],56));
%     imshow(row1matfull);
% end

% let us add sparsity to the dictionary itself
fprintf('*********** THIRD EXPERIMENT ***********\n');
param.modeParam=0;
param.iter=1000;
param.gamma1=5;
param.modeD=1;

tic

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X1=X(:,1:floor(size(X,2)/10));
% X2=X(:,floor(size(X,2)/10):end);
% param.iter=1000;
% tic
% [D model] = mexTrainDL(X1,param);
% t=toc;
% fprintf('time of computation for Dictionary Learning: %f\n',t);
% fprintf('Evaluating cost function...\n');
% alpha=mexLasso(X,D,param);
% 
% % R=mean(0.5*sum((X-D*alpha).^2)+param.lambda*sum(abs(alpha)));
% % fprintf('objective function: %f\n',R);
% tic
% % Then reuse the learned model to retrain a few iterations more.
% param2=param;
% param2.D=D;
% [D model] = mexTrainDL(X2,param2,model);
% %[D] = mexTrainDL(X,param);
% t=toc;
% fprintf('time of computation for Dictionary Learning: %f\n',t);
% fprintf('Evaluating cost function...\n');
% alpha=mexLasso(X,D,param);
% R=mean(0.5*sum((X-D*alpha).^2)+param.lambda*sum(abs(alpha)));
% fprintf('objective function: %f\n',R);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[D] = mexTrainDL(X,param);
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
% subplot(1,3,2);
% ImD=displayPatches(D);
% imagesc(ImD); colormap('gray');
% drawnow;

return;
% fprintf('*********** FOURTH EXPERIMENT ***********\n');
% param.modeParam=0;
% param.iter=1000;
% param.gamma1=0.3;
% param.modeD=3;
% tic
% [D] = mexTrainDL(X,param);
% t=toc;
% fprintf('time of computation for Dictionary Learning: %f\n',t);
% fprintf('Evaluating cost function...\n');
% alpha=mexLasso(X,D,param);
% R=mean(0.5*sum((X-D*alpha).^2)+param.lambda*sum(abs(alpha)));
% fprintf('objective function: %f\n',R);
% tic
% %% Show first 20 2-digit images
% figure('Name','4th exp')
% for i=1:20
%     fprintf('%d. Hello world!\n', i);
%     subplot(4,5,i)
%     row1matfull = full(reshape(alpha(i,:),[],56));
%     imshow(row1matfull);
% end
% sigma=0.0;
% subplot(1,3,3);
% ImD=displayPatches(D);
% imagesc(ImD); colormap('gray');
% drawnow;
