function [W,H,errs] = convnmf_kl_onlyW(V,H,r,win,niter)
% function [W,H,errs,vout] = nmf_kl_con(V,r,varargin)



[n,m] = size(V);

% process arguments
% [win, niter, thresh, norm_w, norm_h, verb, myeps, W0, H0, W, H] = ...
%     parse_opt(varargin, 'win', 1, 'niter', 100, 'thresh', [], ...
%                         'norm_w', 1, 'norm_h', 0, 'verb', 1, ...
%                         'myeps', 1e-20, 'W0', [], 'H0', [], ...
%                         'W', [], 'H', []);

% initialize W based on what we got passed
 W = rand(n,r,win);
norm_w=1; norm_h= 0; verb=1;myeps= 1e-20;
% initialize H based on what we got passed

                    
if norm_w ~= 0
    % normalize W
    W = normalize_W(W,norm_w);
end

if norm_h ~= 0
    % normalize H
    H = normalize_H(H,norm_h);
end

% preallocate matrix of ones
Onm = ones(n,m);

errs = zeros(niter,1);
for t = 1:niter
    % update W if requested
    if update_W
        for k = 0:win-1
            % approximate V
            R = rec_cnmf(W,H,myeps);
            
            % update W
            W(:,:,k+1) = W(:,:,k+1) .* (((V./R)*shift(H,k)') ./ ...
                                        max(Onm*shift(H,k)', myeps));
        end
        
        if norm_w ~= 0
            % normalize columns of W
            W = normalize_W(W,norm_w);
        end
    end
    
    % update reconstruction
    R = rec_cnmf(W,H,myeps);
   
    
    % compute I-divergence
    errs(t) = sum(V(:).*log(V(:)./R(:)) - V(:) + R(:));
    
    % display error if asked
    if verb >= 3
        disp(['nmf_kl_con: iter=' num2str(t) ', err=' num2str(errs(t))]);
    end
    
end

% display error if asked
if verb >= 2
    disp(['nmf_kl_con: final_err=' num2str(errs(t))]);
end

% if we broke early, get rid of extra 0s in the errs vector
errs = errs(1:t);

end

function R = rec_cnmf(W,H,myeps)
% function R = rec_cnmf(W,H,myeps)
%
% Reconstruct a matrix R using Convolutive NMF using W and H matrices.
%

[n, r, win] = size(W);
m = size(H,2);

R = zeros(n,m);
for t = 0:win-1
    R = R + W(:,:,t+1)*shift(H,t);
end

R = max(R,myeps);

end

function O = shift(I, t)
% function O = shift(I, t)
%
% Shifts the columns of an input matrix I by t positions.  
% Zeros are shifted in to new spots.
%

if t < 0
    O = [I(:,-t+1:end) zeros(size(I,1),-t) ];
else
    O = [zeros(size(I,1),t) I(:,1:end-t) ];
end

end
