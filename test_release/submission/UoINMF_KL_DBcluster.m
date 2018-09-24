function [output] = UoINMF_KL_DBcluster(A,params)
%--------[mdlstrct] = UoINMF_KL_DBcluster(A,params)
% =========================================================================
%    The UoINMFcluster algorithm for robust Nonnegative matrix
%    factorization
%
% =========================================================================
% INPUT ARGUMENTS:
% A                         Input nonnegative matrix
% params                    Structure of parameters. See demo file for
%                           configuration. Main parameters are:
% params.k                  Rank k of the factorization
% params.B1                  number of bootstrap samples for selection
% params.B2                  number of bootstrap samples for bagging (estimation)
% =========================================================================
%
% OUTPUT ARGUMENTS:
% output                    Structure with various information.
% =========================================================================
% 04/26/2018, by Shashanka Ubaru. ubaru001@umn.edu.
% =========================================================================

%% parameters for the procedure
B1 = params.B1; %number of bootstrap samples for bagging (estimation)
B2 = params.B2; %number of bootstrap samples for selection
k  = params.k;  %Rank k of the factorization

%%---Parameters for DBSCAN
epsilon=params.epsilon;
MinPts=params.MinPts;

%%----Other parameters
rndfrct = .8; %fraction of data used for Lasso fitting and estimation
rndfrctL = .9; %fraction of data used for selecting set of bases and weights
verb=1; %verbose display of iterations
[n,d]=size(A);

tic
%% main loop over different bootstraps
%%---generate random subsamples and compute NMF
H2=[];  %save all bases
for c = 1:B1
    if verb
        if mod(c,10)==0; disp(c); end
    end
    %the bootstrap sample
    rndsd = randperm(n); rndsd = rndsd(1:round(rndfrctL*n));
    [~,Htemp,~,~] = nmf_kl(A(rndsd,:),k); % get W and H for different k and different bootstraps
    randind(:,c)=rndsd;
    H2=[H2;Htemp];  %--save current bases
end
%% Choose best bases-set by clustering.
%  Uses clustering idea to select best set of bases
H2(all(~any(H2'), 1 ),:) = []; %remove zero bases  :  useless as none of the elem is zero
H2 = normalize_H(H2,2);
output.H_all=H2;

%   DBSCAN clustering
idx=DBSCAN(H2,epsilon,MinPts);
knew=length(unique(idx));

if(knew<3)
    sprintf('Error:DBSCAN returned only one cluster. Change epsilon');
    return;
end
%%% choose centers as new bases
j1=1;
Cen=zeros(knew,d);
for j=1:knew-1
    H1=H2(idx==j,:);
    Cen(j1,:)=mean(H1); %%Mean as center
    j1=j1+1;
end
Cen = normalize_H(Cen,2);
Cen(any(isnan(Cen), 2), :) = [];
output.H=Cen;
k=knew-1;
%% update W based on the best bases H
for c = 1:B1
    rndsd= randind(:,c);
    for l=1:length(rndsd)
        % W(rndsd(l),:,c)=  l1_ls_nonneg(output.H',A(rndsd(l),:)',beta)';
        W(rndsd(l),:,c)= lsqnonneg(output.H',A(rndsd(l),:)')';
    end
end
randind=randind(:,1:end);

% Do the intersection operation
for l=1:n
    [r,c]=find(randind==l);
    for j=1:length(r)  %for each bootstrap sample
        w = find(W(l,:,c(j)));
        if j==1, intw = w; end
        intw = intersect(intw,w); %support is intersection of supports across bootstrap samples
    end
    intWid{l}=intw;
end
output.Wid=intWid;
clear W; clear Htemp; clear intWid;
%% Estimation of weights (Union)
%%--get bootstrap estimates for the support (weights)
%Wnew=zeros(n,k,B2);
for c=1:B2 %number of bootstrap samples for aggregation
    rndsd = randperm(n);rndsd = rndsd(1:round(rndfrct*n));%Btraining data
    for l=1:length(rndsd)
        wind=output.Wid{rndsd(l)};
        if(~isempty(wind))
            zdids = setdiff(1:k,wind);
            %calculate parameters based on reduced model support
            Wnew(l,wind,c)= lsqnonneg(output.H(wind,:)',A(rndsd(l),:)')';
            Wnew(l,zdids,c) = 0;
        else
            Wnew(l,1:k,c) = 0;
        end
    end
    %caclulate error in reconstruction for
    %each bootstrap sample
    err = norm(A(rndsd,:)-Wnew(:,:,c)*output.H,'fro');
    output.err(c) =err;
    randind2(:,c)=rndsd;
end
%%%get bagged estimate of model associated with best regularization parameter
%for each bootstrap sample, find regularization parameter that gave best results
for l=1:n
    [r1,c1]=find(randind2==l);
    for j=1:length(r1)  %for each bootstrap sample
        W_bag(j,:) = Wnew(r1(j),:,c1(j));
    end
    Wopt(l,:)=median(W_bag);
end

output.W = Wopt;%bagged estimates of model parameters

toc
