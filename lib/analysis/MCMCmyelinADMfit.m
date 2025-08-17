function [r, Da, r_std, D_std, noise] = MCMCmyelinADMfit(S, b, delta, Delta, pulsetype, SNR)
%   Code author: Hong-Hsi Lee (orcid: 0000-0002-3663-6559)
%   Massachusetts General Hospital, Harvard Medical School

sz = size(S); % Nmeas by Nsample
Nsample = sz(2);
b = b(:);
delta = delta(:);
Delta = Delta(:);

modelParams = {'r','Da','noise'};
pars0.r     = 1*ones(1,Nsample);
pars0.Da    = 1*ones(1,Nsample);
pars0.noise = 1./SNR.*ones(1,Nsample);

% set up fitting algorithm
fitting                     = [];
% define model parameter name and fitting boundary
fitting.modelParams         = modelParams;
fitting.lb                  = [0, 0, 1e-3];    % lower bound 
fitting.ub                  = [5, 3, 2   ];    % upper bound

% Estimation algorithm setting
fitting.iteration    = 1e4;
fitting.burnin       = 0.1;     % 10% iterations
fitting.thinning     = 5;
fitting.StepSize     = 2;
fitting.Nwalker      = 50;

% define your forward model
modelFWD = @FWD_GD;

% equal weights
weights = [];

y = S;

mcmc_obj    = mcmc;
xPosterior  = mcmc_obj.goodman_weare(y,pars0,weights,fitting,modelFWD,...
    b, delta, Delta, pulsetype);

X     = reshape(xPosterior.r    ,[Nsample, prod(size(xPosterior.r,2:3))]);
X     = rmoutliers(X,2);
r     = median(X, 2);
r_std = std(X, 0, 2);

X     = reshape(xPosterior.Da   ,[Nsample, prod(size(xPosterior.Da,2:3))]);
X     = rmoutliers(X,2);
Da    = median(X, 2);
D_std = std(X, 0, 2);

X     = reshape(xPosterior.noise,[Nsample, prod(size(xPosterior.noise,2:3))]);
% X     = rmoutliers(X,2);
noise = median(X,2);

end

function S = FWD_GD(pars, b, delta, Delta, pulsetype)
r = pars.r;
Da = pars.Da;

if strcmpi(pulsetype, 'wide')
    Dm = @(r, t, del, Del) 1/2*r.^2.*t.^2./del.^2./(Del-del/3) .* ...
        (2*del./t -2 + 2*exp(-Del./t) + 2*exp(-del./t) - exp(-(Del-del)./t) - exp(-(Del+del)./t) );
else
    Dm = @(r, t, del, Del) 1/2*r.^2./(Del+del).*( 1 - exp(-(Del+del)./t) );
end

Dr = Dm(r, r.^2./Da, delta, Delta);
S = sqrt(pi/4./b./abs(Da-Dr)) .* exp(-b.*Dr) .* erf(sqrt(b.*abs(Da-Dr)));

end