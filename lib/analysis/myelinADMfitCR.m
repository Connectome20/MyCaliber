function [r, Da, r_std, D_std] = myelinADMfitCR(S, b, delta, Delta)
%   Code author: Hong-Hsi Lee (orcid: 0000-0002-3663-6559)
%   Massachusetts General Hospital, Harvard Medical School

S = S(:);
b = b(:);
delta = delta(:);
Delta = Delta(:);

q = sqrt(b./(Delta-delta/3));
teff = Delta - delta/3;
texp = Delta + delta;

Nt = 2e2;
Xt = zeros(2,Nt);
CI = zeros(2,2,Nt);
lb = [0 0];
ub = [5 3];
rng(0);
for i = 1:Nt
    x0 = [5*rand, 3*rand];
    options = optimoptions(@lsqnonlin,'Display','off','Algorithm','Levenberg-Marquardt');
    [Xt(:,i),~,residual,~,~,~,jacobian] = lsqnonlin(@(x)costfunction(S, q, teff, texp, x), ...
        x0, lb, ub, options);
    CI(:,:,i) = nlparci(Xt(:,i),residual,'jacobian',jacobian);
end
idx = kmeans(Xt.',3);
Xt = Xt(:,idx==mode(idx));
r = median(Xt(1,:));
[~,I] = min(abs( Xt(1,:)-median(Xt(1,:)) ));
Da = Xt(2,I);

tmp = diff(CI(1,:,:),1,2);
tmp = squeeze(tmp);
if nnz(tmp<100)>1
    tmp = tmp(tmp<100);
end
r_std = median(tmp)/4;

tmp = diff(CI(2,:,:),1,2);
tmp = tmp(~isnan(tmp));
tmp = squeeze(tmp);
if nnz(tmp<100)>1
    tmp = tmp(tmp<100);
end
D_std = median(tmp)/4;

end

function J = costfunction(S, q, teff, texp, X)
r = X(1);
Da = X(2);

rq2sq = r.*q/2.*sqrt(teff./texp);
q2Dat = q.^2.*Da.*teff;
D0tr2 = Da*texp./r.^2;

SX = myelinSMTCG(rq2sq, q2Dat, D0tr2);
J = SX-S;

end

function S = myelinSMTCG(rq2sq, q2Dat, D0tr2)
kmax = 3;
I1 = 0;
for k = 0:kmax
    factor_I1 = (-1)^k/factorial(k)^2*nchoosek(2*k,k) .* rq2sq.^(2*k);
    for j = 0:k
        I1 = I1 + factor_I1 .* ...
            nchoosek(k, j)*(-1)^j .* (gamma(j+1/2)-igamma(j+1/2,q2Dat))./q2Dat.^(j+1/2);
    end
end

pmax = 3;
I2 = 0;
for p = 1:pmax
    for k = 0:kmax
        factor_I2 = exp(-p^2*D0tr2) .* ...
                (-1)^k/factorial(k)/factorial(2*p+k) * nchoosek(2*(p+k),p+k) .* rq2sq.^(2*(p+k));
        for j = 0:(p+k)
            I2 = I2 + factor_I2 .* ...
                nchoosek(p+k,j) * (-1)^j .* (gamma(j+1/2)-igamma(j+1/2,q2Dat))./q2Dat.^(j+1/2);
        end
    end
end
S = (I1+I2)/2;

end

% function b = mynchoosek(n, k)
% b = factorial(n)./factorial(n-k)./factorial(k);
% end