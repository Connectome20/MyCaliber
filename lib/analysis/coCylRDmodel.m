function RD = coCylRDmodel(ri, ro, delta, Delta, D0)
%COCYLRDMODEL    RD of cylindrical shells of finite thickness
%   coCylRDmodel(ri, ro, delta, Delta, D0) produces radial diffusivity of
%   a cylindrical shell with an inner radius ri, an outer radius ro, and an
%   intrinsic diffusivity D0 at pulse width delta and diffusin time Delta.
%
%   Reference:
%   Lebois, A. (2014). Brain microstructure mapping using quantitative and 
%   diffsusion MRI (Doctoral dissertation, Université Paris Sud-Paris XI).
%
%   Code author: Hong-Hsi Lee (orcid: 0000-0002-3663-6559)
%   Massachusetts General Hospital, Harvard Medical School

delta = delta(:);
Delta = Delta(:);

besseljP = @(nu, z) besselj(nu-1, z)/2 - besselj(nu+1, z)/2;
besselyP = @(nu, z) bessely(nu-1, z)/2 - bessely(nu+1, z)/2;
myfun    = @(nu, b, ro, ri) besseljP(nu, b*ro/ri) .* besselyP(nu, b) - ...
                            besselyP(nu, b*ro/ri) .* besseljP(nu, b);

N = 10;
beta = zeros(1,N);
for i = 1:N
    beta(i) = fzero(@(x)myfun(1, x, ro, ri), (i-1)*ri/(ro-ri)*pi + 0.3*pi);
end

roi = ro/ri;
A1m = ri^2/2./beta.^2 .* ...
    ( (beta.^2*roi^2 - 1).*(besselj(1,beta*roi).*besselyP(1,beta) - bessely(1,beta*roi).*besseljP(1,beta)).^2 ...
    - (beta.^2 - 1).*(besselj(1,beta).*besselyP(1,beta) - bessely(1,beta).*besseljP(1,beta)).^2 );
A1m = 1./A1m;

I = ri^6./beta.^2 .* ...
    ( besselyP(1,beta).*(besselj(2,beta*roi)*roi^2 - besselj(2,beta)) ... 
    - besseljP(1,beta).*(bessely(2,beta*roi)*roi^2 - bessely(2,beta)) ).^2;

tc = ri^2/D0;
bardelta = delta/tc;
barDelta = Delta/tc;

RD = ri^4/D0^2/(ro^2-ri^2)*...
    sum( A1m.*I./beta.^4.*(2*beta.^2.*bardelta - 2 ... 
    + 2*exp(-beta.^2.*bardelta) + 2*exp(-beta.^2.*barDelta) ...
    - exp(-beta.^2.*(barDelta+bardelta)) - exp(-beta.^2.*(barDelta-bardelta)) ), 2);

RD = RD./delta.^2./(Delta-delta/3);

end