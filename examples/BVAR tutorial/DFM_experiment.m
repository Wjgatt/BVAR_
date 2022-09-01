% A controlled experiment for a DFM
% see page 96

%=============================================================================================
% We create an artificial database consisting of N = 50 variables with T = 300 generated from
% a dynamic factor model with one lag. The following generates the desired database and
% computes the responses of the endogenous variables to shocks in the factors.
%=============================================================================================
close all; clc;  clear;

addpath ../../cmintools/
addpath ../../bvartools/

% Fix the seed
rng('default'); 
rng(999);

tic;
% Setting the parameters
% AR and Cov factors
Phi = [ 0.75 0.0; 
        0.10 0.8]; 
    
Sigma = [1.0 -0.2; 
        -0.2  2.0]; 
    
Q = chol(Sigma,'lower');

% factor loadings: (50 x 2) matrix with choleski structure
Lambda = [1 0; 0.3 1; 0.5 -1.5; -1.1 -0.5; 1.5 1.5; 5*randn(45,size(Phi,1))];

% persistence and StD of idiosyncratic errors
rho = 0; 
sig = 1;

% preallocating memory
% sample length
T = 301;
f = zeros(T,size(Phi,1)); 
y = zeros(T,size(Lambda,1));
e = zeros(T,size(Lambda,1)); 
i = zeros(T,size(Phi,1));

% Loop
for t = 1 : T-1
    % factor innovations
    i(t+1,:) = randn(size(f,2),1)';
    
    % factors
    f(t+1,:) = (Phi*f(t,:)' + Q*i(t+1,:)')';
    
    % idiosyncratic errors
    e(t+1,:) = (rho*e(t,:)' + sig*randn(size(Lambda,1),1))';
    
    % observed data
    y(t+1,:) = (Lambda*f(t+1,:)' + e(t+1,:)')';
end

f(1,:) = [];
y(1,:) = [];

% Compute true IRFs
hor = 24;
true = iresponse(Phi,eye(size(f,2)),hor,Q);
Lam_ = repmat(Lambda,1,1,size(f,2));
for ff = 1 : size(f,2)
    ir_true(:,:,ff) = Lambda * true(:,:,ff);
end

%=============================================================================================
% Example 29 (Principal Components and scree plot) We extract the first two principal
% components from the artificial data y and plot the first ten eigenvalues (scree plot). The
% commands are:
%=============================================================================================

nfac = 2; 
transf = 0;
[~,pc,~,egv,~] = pc_T(y, nfac, transf);

% scree plot
figure, plot(egv(1:10),'b'); grid 'on';

%=============================================================================================
%% Example 30 (Static factor model) We estimate a static factor model on the artificial
% data y, assuming two factors. The prior for Λ has mean is zero and the variance is 6. We
% plot the estimated factors against the ‘true’ ones (i.e. f). We burn the first 5,000 draws and
% plot one every 20 draws of the remaining draws. The commands are:
%=============================================================================================
tic;
% 2 static factors
nfac = size(f,2);
lags = 0;

% priors
options.priors.F.Lambda.mean = 0;
options.priors.F.Lambda.cov = 6;
options.ndraws = 1000;

% estimation command
[BSFM] = bdfm_(y,lags,nfac,options);

% consider a subset of the draws after the 5000th
index = 5:20:size(BSFM.Phi_draws,3);

% plot estimates and true factors
figure;
for gg=1:nfac
    subplot(nfac,1,gg)
    
    % draws of factors, take all of them
    plot([squeeze(BSFM.f_draws(:,gg,index))],'Color',[0.7 0.7 0.7])
    hold on
    
    % true factors
    plot(f(:,gg),'b','Linewidth',2)
    
    % mean across draws
    plot(mean(BSFM.f_draws(:,gg,index),3),'k','Linewidth',2)
    
end
toc;
%=============================================================================================
%% Example 31 (Dynamic factor model) We estimate a dynamic factor model on the artificial
% data y, assuming two factors and one lag. We again plot the estimated factors against
% the ‘true’ ones (i.e. f). The estimated factors are computed burning the first 5,000 draws
% and considering one every 20 of the remaining draws. The commands are:
% 2 dynamic factors with one lag
%=============================================================================================
nfac = 2;
lags = 1;

% estimation command
[BDFM] = bdfm_(y,lags,nfac);

% consider a subset of draws after the 5000th
% index = 5000:20:size(BDFM.Phi_draws,3);
index = 500:20:size(BDFM.Phi_draws,3);

figure,
for gg=1:nfac
    subplot(nfac,1,gg)
    % draws of factors
    plot([squeeze(BDFM.f_draws(:,gg,index))],'Color',[0.7 0.7 0.7])
    hold on
    % true factors
    plot(f(:,gg),'b','Linewidth',2)
    hold on
    % mean across draws
    plot(mean(BDFM.f_draws(:,gg,index),3),'k','Linewidth',2)
end

toc;
