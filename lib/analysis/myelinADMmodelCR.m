function S = myelinADMmodelCR(r, b, delta, Delta, Da, varargin)
%   Code author: Hong-Hsi Lee (orcid: 0000-0002-3663-6559)
%   Massachusetts General Hospital, Harvard Medical School

b = b(:);
Delta = Delta(:);
delta = delta(:);

q = sqrt(b./(Delta-delta/3));
teff = Delta - delta/3;
texp = Delta + delta;

rq2sq = r.*q/2.*sqrt(teff./texp);
q2Dat = q.^2.*Da.*teff;
D0tr2 = Da.*texp./r.^2;

if strcmpi(varargin{1}, 'fast')
    kmax = 5;    
    pmax = 5;
    [p, j, k] = createindex(kmax,pmax);

    igamma_value = igamma(repmat(j+1/2,size(q2Dat,1),1), repmat(q2Dat,1,size(j,2)));
    I = exp(-p.^2.*D0tr2) .* ...
        (-1).^k./factorial(k)./factorial(2*p+k) .* ...
        factorial(2*(p+k))./factorial(p+k).^2 .* ...
        rq2sq.^(2*(p+k)) .*...
        factorial(p+k) ./ factorial(j) ./ factorial(p+k-j) .* ...
        (-1).^j .* (gamma(j+1/2)-igamma_value)./q2Dat.^(j+1/2);
    S = sum(I,2)/2;

else
    kmax = 3;
    I1 = 0;
    for k = 0:kmax
        factor_I1 = (-1)^k/factorial(k)^2*nchoosek(2*k,k) .* rq2sq.^(2*k);
        for j = 0:k
            I1 = I1 + factor_I1 .* ...
                nchoosek(k, j)*(-1)^j .* (gamma(j+1/2)-igamma(j+1/2,q2Dat))./q2Dat.^(j+1/2);
        end
    end

    p_all = []; k_all = []; j_all = [];
    pmax = 3;
    I2 = 0;
    for p = 1:pmax
        for k = 0:kmax
            factor_I2 = exp(-p^2*D0tr2) .* ...
                    (-1)^k/factorial(k)/factorial(2*p+k) * nchoosek(2*(p+k),p+k) .* rq2sq.^(2*(p+k));
            for j = 0:(p+k)
                I2 = I2 + factor_I2 .*...
                    nchoosek(p+k,j) * (-1)^j .* (gamma(j+1/2)-igamma(j+1/2,q2Dat))./q2Dat.^(j+1/2);
                p_all = cat(2,p_all,p);
                k_all = cat(2,k_all,k);
                j_all = cat(2,j_all,j);
            end
        end
    end

    S = (I1+I2)/2;

end



end


function [p, j, k] = createindex(kmax,pmax)

% sequences_cell = arrayfun(@(k) 0:k, 0:kmax, 'UniformOutput', false);
% j0 = cell2mat(sequences_cell);
% sequences_cell = arrayfun(@(k) k*ones(1,k+1), 0:kmax, 'UniformOutput', false);
% k0 = cell2mat(sequences_cell);

p = [];
j = [];
k = [];
for pj = 0:pmax
    sequences_cell = arrayfun(@(kj) 0:kj, pj:(pj+kmax), 'UniformOutput', false);
    jj = cell2mat(sequences_cell);
    j  = cat(2,j,jj);
    sequences_cell = arrayfun(@(kj) kj*ones(1,kj+1+pj), 0:kmax, 'UniformOutput', false);
    kk = cell2mat(sequences_cell);
    k  = cat(2,k,kk);
    sequences_cell = arrayfun(@(kj) pj*ones(1,kj+1+pj), 0:kmax, 'UniformOutput', false);
    pp = cell2mat(sequences_cell);
    p  = cat(2,p,pp);
end

end