function RD = myelinRDmodel(r, delta, Delta, D0, pulsetype)
%   Code author: Hong-Hsi Lee (orcid: 0000-0002-3663-6559)
%   Massachusetts General Hospital, Harvard Medical School

if strcmpi(pulsetype,'wide')
    Dm = @(r, t, del, Del) 1/2*r.^2.*t.^2./del.^2./(Del-del/3) .* ...
        (2*del./t -2 + 2*exp(-Del./t) + 2*exp(-del./t) - exp(-(Del-del)./t) - exp(-(Del+del)./t) );
else
    Dm = @(r, t, del, Del) r^2/2./(Del+del).*(1-exp(-(Del+del)./t));
end
RD = Dm(r, r.^2/D0, delta, Delta);

end
