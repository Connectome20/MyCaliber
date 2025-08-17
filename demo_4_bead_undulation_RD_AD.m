% ********** Setup the directory on your computer **********
% demo 4: MC simulations of RD in coaxial cylindrical shells of finite 
% thickness with beadings and undulations, 1 layer, impermeable membrane
clear
restoredefaultpath
filePath = matlab.desktop.editor.getActiveFilename;
root0 = fileparts(filePath);
addpath(genpath(fullfile(root0,'lib')));

root = fullfile(root0,'data');
root_cuda = fullfile(root0,'lib','rms');

% project name
projname = 'bead_undulation_RD_AD';

mkdir(fullfile(root,projname));

%% Generate microgeometry

% generate cylinder with beading and undulation
rs = randstick();
abar = 5;       % mean distance between beadings, um
astd = 2.5;     % std of distance between beadings, um
lbar = 5;       % bead width, um
rcsa = 1;       % mean radius of cross-section, um
cv = [0 0.2];   % coefficient of variation of radius = std(r)/mean(r)
w0 = [0 0.5];   % undulation amplitude, um
la = 20;        % undulation wavelength, um
Lz = la*3;      % axon length in z, um
lz = 0.02;      % frustum thickness in z, um
Nz = round(Lz/lz);
% frustum thickness lz should be larger than the step size sqrt(6*D0*dt)

% thickness of each myelin layer, um
% myelin layer thickness lm should be larger than the step size sqrt(6*D0*dt)
lm = 12/1e3;

% pulse width, ms
Td = 6:10;

% width of refocusing RF pulse, ms
tRF180 = 5;

% diffusion time, ms
TD = Td+tRF180;

% b-value, ms/um2
bval = 1:6;

% gradient direction
% 4 directions transverse to cylinder and 1 direction along cylinder
theta = linspace(0, pi, 5); theta = theta(1:end-1); theta = theta(:);
bvec = [cos(theta), sin(theta), 0*theta; 0 0 1];

% # myelin layers
Nm = 1;

% simulation parameters
dt = 1e-6;                      % time of each step, ms
TN = ceil(max(TD+Td)/dt)+100;   % # steps
NPar = 1e5;                     % # random walkers
Din = 0.8;                      % intrinsic diffusivity, um2/ms
threadpb = 256;                 % thread per block for CUDA

seed = 0;
for i = 1:numel(cv)
    for j = 1:numel(w0)
        seed = seed + 1;

        cvi = cv(i);
        w0j = w0(j);

        % center of mass for axon skeleton
        cm = rs.lissajous(Nz,Lz,[w0j 0],[la la],0);

        % caliber variation
        rb = -1*ones(Nz,1);
        seedi = 18;
        rmin = 0.4;    % smallest allowable radius, um
        rdiff = 0.1;   % smallest radius diff btw top and bottom frustums, um
        while any(rb<rmin | abs(rb(1)-rb(end))>rdiff)
            pb = rs.randbeadpos(seedi,Nz,Lz,abar,astd); % bead position, um
            rb = rs.randbeadrad(Nz,Lz,pb,lbar,rcsa,cvi);% radis along z, um
            seedi = seedi + 1;
        end

        target = fullfile(root,projname,sprintf('CoCyl_%04u',seed));
        mkdir(target);
    
        % field of view, um
        xyrange = [cm(:,1)+rb, cm(:,1)-rb, cm(:,2)+rb, cm(:,2)-rb];
        res = 2*(max(abs(xyrange(:)))+Nm*lm)*1.05;

        % circle x coordinate
        xi = cm(:,1)/res + 0.5;
        fileID = fopen(fullfile(target,'phantom_xCir.txt'),'w');
        fprintf(fileID,sprintf('%g\n',xi));
        fclose(fileID);

        % circle y coordinate
        yi = cm(:,2)/res + 0.5;
        fileID = fopen(fullfile(target,'phantom_yCir.txt'),'w');
        fprintf(fileID,sprintf('%g\n',yi));
        fclose(fileID);

        % circle radius
        ri = rb/res;
        fileID = fopen(fullfile(target,'phantom_rCir.txt'),'w');
        fprintf(fileID,sprintf('%g\n',ri));
        fclose(fileID);
        
        % save simulation parameters
        fileID = fopen(fullfile(target,'simParamInput.txt'),'w');
        fprintf(fileID,sprintf('%g\n%u\n%u\n%g\n%u\n%g\n%u\n%g\n%u\n%g\n',...
            dt, TN, NPar, Din, threadpb, ...
            lm/res, Nm, lz/res, Nz, res));
        fclose(fileID);
    
        % save diffusion time and pulse width
        NDelta = numel(TD);
        DELdel = zeros(NDelta,2);
        for iii = 1:numel(TD)
            DELdel(iii,:) = [TD(iii), Td(iii)];
        end
        DELdel = DELdel.';
        DELdel = DELdel(:);
        fid = fopen(fullfile(target,'gradient_NDelta.txt'),'w');
        fprintf(fid,sprintf('%u\n',NDelta));
        fclose(fid);
    
        fid = fopen(fullfile(target,'gradient_DELdel.txt'),'w');
        fprintf(fid,sprintf('%.8f\n',DELdel));
        fclose(fid);
        
        % save b-table
        ig = 0;
        btab = zeros(numel(bval)*size(bvec,1),4);
        for jjj = 1:numel(bval)
            bvalj = bval(jjj);
            for kkk = 1:size(bvec,1)
                ig = ig+1;
                bveck = bvec(kkk,:);
                btab(ig,:) = [bvalj bveck];
            end
        end
        btab = btab.';
        btab = btab(:);
        fid = fopen(fullfile(target,'gradient_Nbtab.txt'),'w');
        fprintf(fid,sprintf('%u\n',numel(bval)*size(bvec,1)));
        fclose(fid);
    
        fid = fopen(fullfile(target,'gradient_btab.txt'),'w');
        fprintf(fid,sprintf('%.8f\n',btab));
        fclose(fid);
    end
end

%% Plot geometry
cv_all = zeros(numel(cv)*numel(w0),1);      % coefficient of variation of radius
w0_all = zeros(numel(cv)*numel(w0),1);      % undulation amplitude, um
seed = 0;
for i = 1:numel(cv)
    for j = 1:numel(w0)
        seed = seed + 1;
        cv_all(seed) = cv(i);
        w0_all(seed) = w0(j);
    end
end

% plot fibers
figure('unit','inch','position',[0 0 4 5]);
for i = 1:4
    subplot(1,4,i);
    target = fullfile(root,projname,sprintf('CoCyl_%04u',i));
    simpara = load(fullfile(target,'simParamInput.txt'));
    res = simpara(10);
    xi = load(fullfile(target,'phantom_xCir.txt'))*res;
    yi = load(fullfile(target,'phantom_yCir.txt'))*res;
    ri = load(fullfile(target,'phantom_rCir.txt'))*res;
    Lz = 60;        % axon length in z, um
    lz = 0.02;      % frustum thickness in z, um
    Nz = round(Lz/lz);

    [X,Y,Z] = cylinder(ri,100);
    X = X+xi;
    Y = Y+yi;
    Z = Z*Lz;

    surf(X,Y,Z,'edgealpha', 0, 'facecolor', 0.99*[1 1 1]);
    camlight
    camlight
    axis off equal
end

%% Create a shell script to run the codes

fileID = fopen(fullfile(root,projname,'job.sh'),'w');
fprintf(fileID,'#!/bin/bash\n');
for i = 1:4
    target = fullfile(root,projname,sprintf('CoCyl_%04u',i));
    fprintf(fileID,sprintf('cd %s\n',target));
    fprintf(fileID,sprintf('cp -a %s .\n',fullfile(root_cuda,'main_PGSE_stringbead_cuda')));
    fprintf(fileID,'./main_PGSE_stringbead_cuda\n');
end
fclose(fileID);

% Open the terminal window in the project folder and run "sh job.sh"

% You may need to open the terminal in the root_cuda folder and compile the 
% CUDA code using "nvcc main_PGSE_stringbead.cu -o main_PGSE_stringbead_cuda"

%% Plot simulation result, RD, [CV(r), w0] = [0,0], [0,0.5], [0.2,0], [0.2,0.5]
cv = [0 0.2];   % coefficient of variation of radius = std(r)/mean(r)
w0 = [0 0.5];   % undulation amplitude, um
cv_all = zeros(numel(cv)*numel(w0),1);
w0_all = zeros(numel(cv)*numel(w0),1);
seed = 0;
for i = 1:numel(cv)
    for j = 1:numel(w0)
        seed = seed + 1;
        cv_all(seed) = cv(i);
        w0_all(seed) = w0(j);
    end
end

cv_un = unique(cv_all(:));
w0_un = unique(w0_all(:));

figure('unit','inch','position',[0 0 10 5]);
clear h ht lgtxt
cmap = colormap('lines');
mk = {'o','v','s','d'};

% RD of narrow-pulse
subplot(121)
hold on;
for j = 1: numel(cv_all)
    % simulation
    rms = simul3DcoCyl_cuda_pgse_bvec(fullfile(root,projname,sprintf('CoCyl_%04u',j)),'bead');
    theta = linspace(0, pi, 5); theta = theta(1:end-1); theta = theta(:);
    bvec = [cos(theta), sin(theta), 0*theta];
    [~, RD] = rms.akc_mom(bvec);
    RD = mean(RD, 2);
    
    % cylindrical shell radius at the middle thickness, um
    r = rms.rCir + rms.Nm.*rms.lm - rms.lm/2;

    % intrinsic diffusivity, um2/ms
    D0 = rms.Din;
    
    % theory
    Dm = @(ri,t) ri.^2/2./t.*(1-exp(-t./(ri.^2/D0)));
    t_plot = 0.01:0.01:100;
    D_plot = 0; f = r/sum(r);
    for i = 1:numel(r)
        D_plot = D_plot + f(i)*Dm(r(i),t_plot);
    end

    Icv = find(cv_all(j)==cv_un);
    Iw0 = find(w0_all(j)==w0_un);
    plot(1./rms.TD(1:20:end),RD(1:20:end),mk{Iw0},'markersize',6,'linewidth',1, 'color', cmap(Icv,:)); 
    plot(1./t_plot, D_plot, '-', 'linewidth', 1, 'color', cmap(Icv,:));
end

for j = 1:numel(w0_un)
    h(j) = plot(-1, -1,mk{j},'markersize',6,'linewidth',1,'Color','k');
    lgtxt{j} = sprintf('$w_0$=%.1f $\\mu$m', w0_un(j));
end

for j = 1:numel(cv_un)
    h(j+numel(w0_un)) = plot(-1, -1,'-','markersize',10,'linewidth',1,'Color',cmap(j,:));
    lgtxt{j+numel(w0_un)} = sprintf('CV($r$)=%.1f', cv_un(j));
end

xlim([0 0.5]); ylim([0 0.25]);
box on; grid on;
set(gca,'fontsize',12);
pbaspect([1 1 1]);
ht = plot(-1, -1, 'k-', 'linewidth',1);
lgtxt{end+1} = 'theory';
legend([h ht],lgtxt,'Interpreter','latex','fontsize',14, ...
    'location','northwest','box','off');
xlabel('$1/t$, ms$^{-1}$','Interpreter','latex','FontSize',20);
ylabel('$D^\perp(t)$, $\mu$m$^2$/ms','Interpreter','latex','FontSize',20);
title('narrow pulse','interpreter','latex','fontsize',20);

% RD of wide pulse
subplot(122);
clear h lgtxt hempty
r_mom = zeros(3,1);
n_dir = 5;
for j = 1:numel(cv_all)
    % simulation
    rms = simul3DcoCyl_cuda_pgse_bvec(fullfile(root,projname,sprintf('CoCyl_%04u',j)),'bead');
    sig = rms.sig;
    bval = unique(rms.bval);
    A = [-bval, 1/6*bval.^2, bval.^(3:6)];
    RD = zeros(rms.NDel,1);
    for i = 1:rms.NDel
        sigi = sig(:,i);
        sigi = reshape(sigi, n_dir, []);
        sigi = mean(sigi(1:4,:),1);

        sigi = max(sigi(:),eps);
        blist = 1:6;
        X = A(blist,:)\log(sigi(blist));
        RD(i) = X(1);
    end
    
    % cylindrical shell radius at the middle thickness, um
    r = rms.rCir + (1:rms.Nm)*rms.lm - rms.lm/2;

    % inner radius, um
    ri = rms.rCir;

    % outer radius, um
    ro = rms.rCir + rms.lm;
    
    % effective radius, um
    n = 1;
    rm = ( sum(r.^(n+1))/sum(r) ).^(1/n);
    r_mom(j) = rm;
    
    D0 = rms.Din;       % intrinsic diffusivity, um2/ms
    tm = rm^2/D0;       % correlation time, ms
    delta = rms.del;    % pulse width, ms
    Delta = rms.Del;    % diffusion time, ms
    tRF180 = 5;         % width of refocusing RF pulse, ms 
    
    % pulse width for plotting, ms
    delta_plot = 1:0.1:100; 

    % diffusion time for plotting, ms
    Delta_plot = delta_plot+tRF180;

    f = r/sum(r);
    D_plot = 0;         % theory
    D_plot_CR = 0;      % approximate solution from Canales-Rodriguez et al. 2025
    for i = 1:numel(r)
        D_plot = D_plot + f(i)*myelinRDmodel(r(i), delta_plot, Delta_plot, D0, 'wide');
        D_plot_CR = D_plot_CR + f(i)*myelinRDmodel(r(i), delta_plot, Delta_plot, D0, 'narrow');
    end
    
    Icv = find(cv_all(j)==cv_un);
    Iw0 = find(w0_all(j)==w0_un);
    hold on;
    plot(1./delta./(Delta-delta/3), RD, mk{Iw0}, 'color', cmap(Icv,:),'markersize',6,'linewidth',1);
    
    plot(1./delta_plot./(Delta_plot-delta_plot/3), D_plot, 'color', cmap(Icv,:),'linewidth',1);
    plot(1./delta_plot./(Delta_plot-delta_plot/3), D_plot_CR, '--', 'color', cmap(Icv,:),'linewidth',1);
end

for j = 1:numel(w0_un)
    h(j) = plot(-1, -1,mk{j},'markersize',6,'linewidth',1,'Color','k');
    lgtxt{j} = sprintf('$w_0$=%.1f $\\mu$m', w0_un(j));
end

for j = 1:numel(cv_un)
    h(j+numel(w0_un)) = plot(-1, -1,'-','markersize',10,'linewidth',1,'Color',cmap(j,:));
    lgtxt{j+numel(w0_un)} = sprintf('CV($r$)=%.1f', cv_un(j));
end

ht = plot(-1, -1, 'k-', 'linewidth',1);
ht_CR = plot(-1, -1, 'k--', 'linewidth', 1);
lgtxt{end+1} = 'theory';
lgtxt{end+1} = 'Canales-Rodr\''iguez et al.';
xlim([0 0.02]); ylim([0 0.045]);
yticks(0:0.01:0.05)
box on; grid on;
set(gca,'fontsize',12);
pbaspect([1 1 1]);
legend([h ht ht_CR],lgtxt,'Interpreter','latex','fontsize',14,...
    'location','northwest','box','off');
xlabel('$1/[\delta\cdot(\Delta-\delta/3)]$, ms$^{-2}$','Interpreter','latex','FontSize',20);
ylabel('$D^\perp(\Delta,\delta)$, $\mu$m$^2$/ms','Interpreter','latex','FontSize',20);
title('wide pulse','interpreter','latex','fontsize',20);


%% Plot simulation result, AD, [CV(r), w0] = [0,0], [0,0.5], [0.2,0], [0.2,0.5]
rcsa = 1;       % mean radius of cross-section, um
cv = [0 0.2];   % coefficient of variation of radius = std(r)/mean(r)
w0 = [0 0.5];   % undulation amplitude, um
cv_all = zeros(numel(cv)*numel(w0),1);
w0_all = zeros(numel(cv)*numel(w0),1);
seed = 0;
for i = 1:numel(cv)
    for j = 1:numel(w0)
        seed = seed + 1;
        cv_all(seed) = cv(i);
        w0_all(seed) = w0(j);
    end
end

cv_un = unique(cv_all(:));
w0_un = unique(w0_all(:));

figure('unit','inch','position',[0 0 10 5]);
clear h ht lgtxt
cmap = colormap('lines');
mk = {'o','v','s','d'};

% AD of narrow-pulse
subplot(121)
hold on;
for j = 1: numel(cv_all)
    % simulation
    rms = simul3DcoCyl_cuda_pgse_bvec(fullfile(root,projname,sprintf('CoCyl_%04u',j)),'bead');
    bvec = [0 0 1];
    [~, AD] = rms.akc_mom(bvec);
    SV = 2/rms.lm;                  % surface-to-volume fraction, 1/um
    ds = sqrt(6*rms.Din*rms.dtime); % step size, um
    dD = rms.Din*3/16*SV*ds;        % rejection sampling term, um2/ms
    
    Icv = find(cv_all(j)==cv_un);
    Iw0 = find(w0_all(j)==w0_un);
    plot(rms.TD(2:40:end),AD(2:40:end)+dD,mk{Iw0},'markersize',6,'linewidth',1, 'color', cmap(Icv,:));
end

for j = 1:numel(w0_un)
    h(j) = plot(-1, -1,mk{j},'markersize',6,'linewidth',1,'Color','k');
    lgtxt{j} = sprintf('$w_0$=%.1f $\\mu$m', w0_un(j));
end

for j = 1:numel(cv_un)
    h(j+numel(w0_un)) = plot(-1, -1,'-','markersize',10,'linewidth',1,'Color',cmap(j,:));
    lgtxt{j+numel(w0_un)} = sprintf('CV($r$)=%.1f', cv_un(j));
end
ylim([0.7 0.9]);
box on; grid on;
set(gca,'fontsize',12);
pbaspect([1 1 1]);
legend(h,lgtxt,'Interpreter','latex','fontsize',14, ...
    'location','northwest','box','off');
xlabel('$t$, ms','Interpreter','latex','FontSize',20);
ylabel('$D^\parallel(t)$, $\mu$m$^2$/ms','Interpreter','latex','FontSize',20);
title('narrow pulse','interpreter','latex','fontsize',20);

% AD of wide-pulse
subplot(122);
clear h lgtxt hempty
r_fit = zeros(3,1);
r_mom = zeros(3,1);
n_dir = 5;
for j = 1:numel(cv_all)
    % simulation
    rms = simul3DcoCyl_cuda_pgse_bvec(fullfile(root,projname,sprintf('CoCyl_%04u',j)),'bead');
    sig = rms.sig;
    bval = unique(rms.bval);
    A = [-bval, 1/6*bval.^2, bval.^(3:6)];
    AD = zeros(rms.NDel,1);
    for i = 1:rms.NDel
        sigi = sig(:,i);
        sigi = reshape(sigi, n_dir, []);
        sigi = sigi(5,:);

        sigi = max(sigi(:),eps);
        blist = 1:6;
        X = A(blist,:)\log(sigi(blist));
        AD(i) = X(1);
    end
    
    delta = rms.del;                % pulse width, ms
    Delta = rms.Del;                % diffusion time, ms

    SV = 2/rms.lm;                  % surface-to-volume fraction, 1/um
    ds = sqrt(6*rms.Din*rms.dtime); % step size, um
    dD = rms.Din*3/16*SV*ds;        % rejection sampling term, um2/ms
    
    Icv = find(cv_all(j)==cv_un);
    Iw0 = find(w0_all(j)==w0_un);
    hold on;
    plot((Delta-delta/3), AD+dD, mk{Iw0}, 'color', cmap(Icv,:),'markersize',6,'linewidth',1);
end

for j = 1:numel(w0_un)
    h(j) = plot(-1, -1,mk{j},'markersize',6,'linewidth',1,'Color','k');
    lgtxt{j} = sprintf('$w_0$=%.1f $\\mu$m', w0_un(j));
end

for j = 1:numel(cv_un)
    h(j+numel(w0_un)) = plot(-1, -1,'-','markersize',10,'linewidth',1,'Color',cmap(j,:));
    lgtxt{j+numel(w0_un)} = sprintf('CV($r$)=%.1f', cv_un(j));
end
xlim([8 12]); ylim([0.7 0.9])
box on; grid on;
set(gca,'fontsize',12);
pbaspect([1 1 1]);
hg = legend(h,lgtxt,'Interpreter','latex','fontsize',14,...
    'location','northwest','box','off');
xlabel('$\Delta-\delta/3$, ms','Interpreter','latex','FontSize',20);
ylabel('$D^\parallel(\Delta,\delta)$, $\mu$m$^2$/ms','Interpreter','latex','FontSize',20);
title('wide pulse','interpreter','latex','fontsize',20);


