function r_min = myelinADMrmin(bmax, delta, Delta, D0, Da, za, SNR, Nav, pulsetype)
%   Code author: Hong-Hsi Lee (orcid: 0000-0002-3663-6559)
%   Massachusetts General Hospital, Harvard Medical School

sigma_bar = za/SNR/sqrt(Nav);

if strcmpi(pulsetype, 'wide')
    Dm = @(r, t, del, Del) 1/2*r.^2.*t.^2./del.^2./(Del-del/3) .* ...
        (2*del./t -2 + 2*exp(-Del./t) + 2*exp(-del./t) - exp(-(Del-del)./t) - exp(-(Del+del)./t) );
else
    Dm = @(r, t, del, Del) 1/2*r.^2./(Del+del) .* (1 - exp(-(Del+del)./t));
end

J = @(r) bmax*Dm(r, r.^2/D0, delta, Delta) .* ...
    sqrt(pi/4/bmax./(Da-Dm(r, r.^2/D0, delta, Delta))) .* ...
    erf(sqrt(bmax.*(Da-Dm(r, r.^2/D0, delta, Delta)))) - sigma_bar;

% Dm = @(r, t, del, Del) r.^2.*t./del./(Del-del/3);
% 
% J = @(r) bmax*Dm(r, r.^2/D0, delta, Delta) .* ...
%     sqrt(pi/4/bmax./Da) .* ...
%     erf(sqrt(bmax.*Da)) - sigma_bar;

if isinf(SNR)
    r_min = 0;
else

    rinit = 0.1:0.1:1;
    r_min = zeros(numel(rinit), 1);
    for i = 1:numel(rinit)
        r_min(i) = fzero(J, rinit(i));
    end
    r_min = min(r_min(r_min>0));

end

end
