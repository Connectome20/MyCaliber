function S = myelinADMmodel(r, b, delta, Delta, Da, pulsetype)
%   Code author: Hong-Hsi Lee (orcid: 0000-0002-3663-6559)
%   Massachusetts General Hospital, Harvard Medical School

if strcmpi(pulsetype, 'wide')
    Dm = @(r, t, del, Del) 1/2*r.^2.*t.^2./del.^2./(Del-del/3) .* ...
        (2*del./t -2 + 2*exp(-Del./t) + 2*exp(-del./t) - exp(-(Del-del)./t) - exp(-(Del+del)./t) );
else
    Dm = @(r, t, del, Del) 1/2*r.^2./(Del+del).*( 1 - exp(-(Del+del)./t) );
end
DX = Dm(r, r.^2./Da, delta, Delta);
S = sqrt(pi/4./b/abs(Da-DX)) .* exp(-b*DX) .* erf(sqrt(b*abs(Da-DX)));

end
