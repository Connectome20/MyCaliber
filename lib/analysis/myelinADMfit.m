function [r, Da, r_std, D_std] = myelinADMfit(S, b, delta, Delta, pulsetype)
%   Code author: Hong-Hsi Lee (orcid: 0000-0002-3663-6559)
%   Massachusetts General Hospital, Harvard Medical School

S = S(:);
b = b(:);
delta = delta(:);
Delta = Delta(:);

Nt = 2e2;
Xt = zeros(2,Nt);
% CI = zeros(2,2,Nt);
se = zeros(2,Nt);
lb = [0 0];
ub = [5 3];
rng(0);
for i = 1:Nt
    x0 = [5*rand, 3*rand];
    options = optimoptions(@lsqnonlin,'Display','off','Algorithm','Levenberg-Marquardt');
    [Xt(:,i),resnorm,residual,~,~,~,jacobian] = lsqnonlin(@(x)costfunction(S, b, delta, Delta, pulsetype, x), ...
        x0, lb, ub, options);
    CI(:,:,i) = nlparci(Xt(:,i),residual,'jacobian',jacobian);
    % % Residual variance
    % n = length(residual);
    % p = length(Xt(:,i));
    % sigma2 = resnorm / (n - p);
    % % Covariance and standard errors
    % covB = sigma2 * inv(jacobian' * jacobian);
    % se(:,i) = sqrt(diag(covB));
end
idx = kmeans(Xt.',3);
Xt = Xt(:,idx==mode(idx));
r = median(Xt(1,:));
[~,I] = min(abs( Xt(1,:)-r ));
Da = Xt(2,I);

tmp = diff(CI(1,:,:),1,2);
tmp = squeeze(tmp);
tmp = tmp(~isnan(tmp));
if nnz(tmp<100)>1
    tmp = tmp(tmp<100);
end
r_std = median(tmp)/4;
% tmp = se(1,:);
% tmp = tmp(~isnan(tmp(:)));
% if nnz(tmp<100)>1
%     tmp = tmp(tmp<100);
% end
% r_std = median(tmp);

tmp = diff(CI(2,:,:),1,2);
tmp = tmp(~isnan(tmp));
tmp = squeeze(tmp);
tmp = tmp(~isnan(tmp));
if nnz(tmp<100)>1
    tmp = tmp(tmp<100);
end
D_std = median(tmp)/4;
% tmp = se(2,:);
% tmp = tmp(~isnan(tmp(:)));
% if nnz(tmp<100)>1
%     tmp = tmp(tmp<100);
% end
% D_std = median(tmp);

end

function J = costfunction(S, b, delta, Delta, pulsetype, X)
r = X(1);
Da = X(2);

if strcmpi(pulsetype, 'wide')
    Dm = @(r, t, del, Del) 1/2*r.^2.*t.^2./del.^2./(Del-del/3) .* ...
        (2*del./t -2 + 2*exp(-Del./t) + 2*exp(-del./t) - exp(-(Del-del)./t) - exp(-(Del+del)./t) );
else
    Dm = @(r, t, del, Del) 1/2*r.^2./(Del+del).*( 1 - exp(-(Del+del)./t) );
end

DX = Dm(r, r.^2./Da, delta, Delta);
SX = sqrt(pi/4./b./abs(Da-DX)) .* exp(-b*DX) .* erf(sqrt(b.*abs(Da-DX)));
J = S-SX;

end
