function r = myelinRDfit(D, delta, Delta, D0, pulsetype)
%   Code author: Hong-Hsi Lee (orcid: 0000-0002-3663-6559)
%   Massachusetts General Hospital, Harvard Medical School

D = D(:);
delta = delta(:);
Delta = Delta(:);

Nt = 1e1;
Xt = zeros(2,Nt);
rng(0);
for i = 1:Nt
    x0 = 5*rand;
    options = optimoptions(@lsqnonlin,'Display','off','Algorithm','Levenberg-Marquardt');
    Xt(:,i) = lsqnonlin(@(x)costfunction(D, delta, Delta, D0, x, pulsetype), x0, 0, 5, options);
end
idx = kmeans(Xt.',3);
Xt = Xt(:,idx==mode(idx));
r(1) = median(Xt(1,:));

end

function J = costfunction(D, delta, Delta, D0, r, pulsetype)

if strcmpi(pulsetype,'wide')
    Dm = @(r, t, del, Del) 1/2*r.^2.*t.^2./del.^2./(Del-del/3) .* ...
        (2*del./t -2 + 2*exp(-Del./t) + 2*exp(-del./t) - exp(-(Del-del)./t) - exp(-(Del+del)./t) );
else
    Dm = @(r, t, del, Del) r^2/2./(Del+del).*(1-exp(-(Del+del)./t));
end
DX = Dm(r, r.^2/D0, delta, Delta);
J = DX-D;

end
