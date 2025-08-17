function [tnp, Dnp, Swp] = cylinderSurfaceSimulation(r, delta, Delta, bval, bvec, D0, dt, Np)
%CYLINDERSURFACESIMULATION    Diffusion simulation on a clyindrical surface
%   [t, D, S] = cylinderSurfaceSimulation(r, del, Del, b, g, D0, dt, N)
%   produces diffusion time t (ms) and radial diffusivity D (um2/ms) of 
%   narrow-pulse sequence and signal S of wide-pulse sequence on a 
%   cylindrical surface of radius r (um) and intrinsic diffusivity D0 
%   (um2/ms). 
% 
%   The wide-pulse sequence has a pulse width del (ms), diffusion time Del 
%   (ms), b-value b (ms/um2), and gradient directions g. The simulation has
%   a time step dt (ms) and N random walkers.
%
%   Code author: Hong-Hsi Lee (orcid: 0000-0002-3663-6559)
%   Massachusetts General Hospital, Harvard Medical School

Td = single(delta);
TD = single(Delta);
dt = single(dt);
Nt = ceil(max(TD+Td)/dt);
Np = single(Np);
dx = sqrt(2*D0*dt);

% narrow pulse simulation
dth = single(dx/r);
thi = 2*pi*rand(Np, 1, 'single');
xi = r*cos(thi);
yi = r*sin(thi);
th = dth * ( 2*(rand(Np, Nt, 'single')>0.5) - 1);
th = cumsum(th, 2);
xt = r*cos(th+thi);
yt = r*sin(th+thi);
x2 = sum((xt-xi).^2,1)/Np;
y2 = sum((yt-yi).^2,1)/Np;
tnp = (1:Nt)*dt;
Dnp = (x2+y2)/2/2./tnp;

zt = dx  * ( 2*(rand(Np, Nt, 'single')>0.5) - 1);
zt = cumsum(zt, 2);

Swp = zeros(numel(TD),numel(bval),size(bvec,1),'single');
% wide pulse signal calculation
for i = 1:numel(TD)
    TDi = TD(i);
    Tdi = Td(i);
    gt = zeros(1,Nt,'single');
    n1 = round(Tdi/dt);
    n2 = round(TDi/dt);
    gt(1:n1) = 1;
    gt(n2+1:n1+n2) = -1;
    gt = repmat(gt, Np, 1);

    for j = 1:numel(bval)
        bvalj = bval(j);
        gj = sqrt(bvalj/Tdi^2/(TDi-Tdi/3));
        for k = 1:size(bvec, 1)
            bveck = bvec(k,:);
            phi = sum(gj*gt.*(xt*bveck(1) + yt*bveck(2) + zt*bveck(3))*dt, 2);
            Swp(i,j,k) = mean(cos(phi));
        end
    end
end

end