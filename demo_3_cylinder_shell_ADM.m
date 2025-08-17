% ********** Setup the directory on your computer **********
% demo 3: MC simulations of SMT in coaxial cylindrical shells of finite 
% thickness, permeable membrane
clear
restoredefaultpath
filePath = matlab.desktop.editor.getActiveFilename;
root0 = fileparts(filePath);
addpath(genpath(fullfile(root0,'lib')));

root = fullfile(root0,'data');
root_cuda = fullfile(root0,'lib','rms');

% project name
projname = 'coaxial_cyliner_shell_ADM';

mkdir(fullfile(root,projname));

%% Generate packing, permeable membrane

% inner radius, um
r = 0.1:0.1:5;

% thickness of each myelin layer, um
% myelin layer thickness lm should be larger than the step size sqrt(4*D0*dt)
lm = 12/1e3;

% # myelin layers
C0 = 0.35/lm;
C1 = 0.006/lm;
C2 = 0.024/lm;
Nm = round(C0 + C1*2*r + C2*log(2*r));

% permeability, um/ms
kappa = 6.7e-3;

% pulse width, ms
Td = 6;

% diffusion time, ms
TD = 13;

% b-value, ms/um2
bval = [50 350 800 1500 2400 3450 4750 6000]/1e3;

% gradient direction, 64 directions per b-shell
% requirement: MRtrix3
bvec = dirgen(64);

% simulation parameters
dt = 1e-6;                      % time step, ms
TN = ceil(max(TD+Td)/dt)+100;   % # steps
NPar = 1e4;                     % # random walkers
D0 = 0.8;                       % intrinsic diffusivity, um2/ms
threadpb = 256;                 % thread per block for CUDA

seed = 0;
for i = 1:numel(r)
    seed = seed + 1;
    target = fullfile(root,projname,sprintf('CoCyl_%04u',seed));
    mkdir(target);

    % field of view, um
    res = 2*(r(i)+Nm(i)*lm)*1.05;

    % save simulation parameters
    fileID = fopen(fullfile(target,'simParamInput.txt'),'w');
    fprintf(fileID,sprintf('%g\n%u\n%u\n%g\n%g\n%u\n%g\n%g\n%u\n%g\n',...
        dt, TN, NPar, D0, kappa, threadpb, ...
        r(i)/res, lm/res, Nm(i), res));
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

%% Create a shell script to run the codes

fileID = fopen(fullfile(root,projname,'job.sh'),'w');
fprintf(fileID,'#!/bin/bash\n');
for j = 1:50
    target = fullfile(root,projname,sprintf('CoCyl_%04u',j));
    fprintf(fileID,sprintf('cd %s\n',target));
    fprintf(fileID,sprintf('cp -a %s .\n',fullfile(root_cuda,'main_PGSE_perm_cuda')));
    fprintf(fileID,'./main_PGSE_perm_cuda\n');
end
fclose(fileID);

% Open the terminal window in the project folder and run "sh job.sh"

% You may need to open the terminal in the root_cuda folder and compile the 
% CUDA code using "nvcc main_PGSE_perm.cu -o main_PGSE_perm_cuda"

%% Plot simulation results
Nr = 50;                    % # cylinder radius
Nbval = 8;                  % # b-value
Nbvec = 64;                 % # gradient direction

% direction dependent diffusion signals
S_dir = zeros(Nr, Nbval, Nbvec);
for i = 1:Nr
    rms = simul3DcoCyl_cuda_pgse_bvec(fullfile(root,projname,sprintf('CoCyl_%04u',i)));
    sigi = rms.sig;
    bval = unique(rms.bval);
    sigi = reshape(sigi,Nbvec,[]);
    S_dir(i,:,:) = sigi.';
end

% effective radius of the n-th order, um
% n = 0, <r>                mean radius
% n = 1, <r^2>/<r>          volume weighted averaged radius
% n = 2, (<r^3>/<r>)^(1/2)  effective radius for narrow-pulse
% n = 4, (<r^5>/<r>)^(1/4)  effective radius for wide-pulse
n = [0 1 2 4];
r_mom = zeros(Nr, numel(n));
vol = zeros(Nr, 1);         % ~volume, um2
Nm  = zeros(Nr, 1);         % # myelin layer
for i = 1:Nr
    rms = simul3DcoCyl_cuda_pgse_bvec(fullfile(root,projname,sprintf('CoCyl_%04u',i)));

    % cylindrical shell radius at the middle thickness, um
    r  = rms.rCir + (1:rms.Nm - 1/2)*rms.lm;

    % inner radius, um
    ri = rms.rCir;

    % outer radius, um
    ro = rms.rCir + rms.Nm*rms.lm;

    vol(i) = pi*(ro^2 - ri^2);
    Nm(i)  = rms.Nm;
    for k = 1:numel(n)
        ni = n(k);
        if ni==0
            r_mom(i,k) = sum(r);
        else
            r_mom(i,k) = sum(r.^(ni+1));
        end
    end
end

%% MyeCaliber
% Gamma distribution for axon radius
ri_mean = (0.25:0.05:1).';      % mean, um
ri_var  = (ri_mean/2).^2;       % variance, um2
b = ri_var./ri_mean;            % scale parameter, um
a = ri_mean./b;                 % shape parameter
x = (0.1:0.1:5).';              % sampled axon radius, um

Nbval = 8;                      % # b-value
Nbvec = 64;                     % # gradient direction per b-shell
SNR = [Inf 50 20 10];           % signal-to-noise ratio for Rician noise

% fitted parameters: radius, axial diffusivity, their standard deviations
r_fit = zeros(numel(ri_mean), numel(SNR), 2);
D_fit = zeros(numel(ri_mean), numel(SNR), 2);
r_std = zeros(numel(ri_mean), numel(SNR), 2);
D_std = zeros(numel(ri_mean), numel(SNR), 2);

% spherical mean signal with noise for the fitting
S_fit = zeros(numel(ri_mean), numel(SNR), Nbval);

% model selection
% narrow: narrow-pulse solution (Canales-Rodriguez et al. 2005)
% wide: wide-pulse solution
pulsetype = {'narrow', 'wide'};

tic;
for j = 1:numel(ri_mean)
    % direction-dependent diffusion signals from axons with radii in Gamma 
    % distribution
    Ni = gampdf(x, a(j), b(j));
    Si = sum(Ni.*vol.*S_dir)/sum(Ni.*vol);
    Si = squeeze(Si);
    for i = 1:numel(SNR)
        % apply Rician noise to direction-dependent diffusion signals
        sigma = 1/SNR(i);
        Sj = abs( Si + sigma*randn(Nbval, Nbvec) + 1j*sigma*randn(Nbval, Nbvec) );

        % calculate spherical mean signal
        Sj = mean(Sj, 2);

        % Rician noise floor correction
        Sj = sqrt(max(Sj.^2-sigma^2, 0));
        
        % model fitting
        S_fit(j,i,:) = Sj;
        for k = 1:numel(pulsetype)
            [r_fit(j,i,k), D_fit(j,i,k), r_std(j,i,k), D_std(j,i,k)] = myelinADMfit(Sj, bval, rms.del, rms.Del, pulsetype{k});
        end
    end
end
toc;

% effective radius of the n-th order with axon radius distribution, um
% n = 0, <r>                mean radius
% n = 1, <r^2>/<r>          volume weighted averaged radius
% n = 2, (<r^3>/<r>)^(1/2)  effective radius for narrow-pulse
% n = 4, (<r^5>/<r>)^(1/4)  effective radius for wide-pulse
n = [0 1 2 4];
r_eff = zeros(numel(ri_mean), numel(n));
for j = 1:numel(ri_mean)
    Ni = gampdf(x, a(j), b(j));
    for i = 1:numel(n)
        ni = n(i);
        if ni==0
            r_eff(j,i) =   sum(Ni.*r_mom(:,i))/sum(Ni.*Nm);
        else
            I = find(n==0);
            r_eff(j,i) = ( sum(Ni.*r_mom(:,i))/sum(Ni.*r_mom(:,I)) ).^(1/ni);
        end
    end
end

%% Canales-Rodriguez et al. 2025, full solution fitting

% build training data for random forest regression
rCR = 0.01:0.01:3;          % axon radius, um
DCR = 0.01:0.01:1.6;        % intrinsic diffusivity, um2/ms
% normalized spherical mean signal
SCR = zeros(numel(bval),numel(rCR),numel(DCR));
option = 'fast';
tic;
if strcmpi(option, 'slow')
    for i = 1:numel(bval)
        SCR(i,:,:) = myelinADMmodelCR(rCR(:), bval(i), rms.del, rms.Del, DCR(:).','slow');
    end
else
    for i = 1:numel(bval)
        for j = 1:numel(DCR)
            SCR(i,:,j) = myelinADMmodelCR(rCR(:), bval(i), rms.del, rms.Del, DCR(j),'fast');
        end
    end
end
toc;

% random forest regression for each SNR
X = permute(SCR, [2, 3, 1]);
X = reshape(X, [], 8);
Y = repmat(rCR(:),numel(DCR),1);
r_fit_CR = zeros(numel(ri_mean), numel(SNR));
tic;
for i = 1:numel(SNR)
    % apply Rician noise
    sigma = 1/SNR(i);
    Xi = abs( X + sigma*randn([size(X), Nbvec]) + 1j*sigma*randn([size(X), Nbvec]) );
    Xi = mean(Xi, 3);

    % Rician noise floor correction
    Xi = sqrt(max(Xi.^2-sigma^2, 0));

    % model fitting
    Mdl = TreeBagger(10,Xi,Y,'Method','regression','OOBPrediction','On');
    for j = 1:numel(ri_mean)
        r_fit_CR(j,i) = predict(Mdl,squeeze(S_fit(j,i,:)).');
    end
end
toc;

%% Plot normalized spherical mean signal
figure('unit','inch','position',[0 0 18 4]);
cmap = colormap('lines');
mk = {'o','v','s'};
for i = 1:numel(SNR)
    subplot(1,numel(SNR),i)
    clear h lgtxt
    hold on;
    list = [1 numel(ri_mean)];
    for j = 1:numel(list)
        Si = S_fit(list(j),i,:);
        Si = squeeze(Si);
        h(j) = plot(1./sqrt(bval), Si, mk{j}, 'markersize', 6, 'linewidth', 1, 'color', cmap(j,:));
        bval_plot = 1./linspace(0.01, 5, 1000).^2;
        S_WP = myelinADMmodel(r_fit(list(j),i,2), bval_plot, rms.del, rms.Del, D_fit(list(j),i,2), 'wide');
        plot(1./sqrt(bval_plot), S_WP, '-' , 'linewidth', 1, 'color', cmap(j,:));
        lgtxt{j} = sprintf('$2\\bar{r}_i=$%.1f $\\mu$m', ri_mean(list(j))*2);
    end
    h(3) = plot(-1, -1, 'k-', 'linewidth', 1);
    lgtxt{3} = 'fitting';
    xlim([0 2]);
    ylim([0 1]);
    pbaspect([1 1 1]);
    xlabel('$1/\sqrt{b}$, $\mu$m$\cdot$ms$^{-1/2}$', 'interpreter', 'latex', 'fontsize', 20)
    if i==1, ylabel('$\bar{S}(b)$', 'interpreter', 'latex', 'fontsize', 20); end
    title(sprintf('SNR=%u',SNR(i)),'interpreter','latex','fontsize',20)
    box on; grid on;
    legend(h, lgtxt,'interpreter','latex','fontsize',16,'box','off','location','southeast')
end

%% Resolution limit
D0 = rms.Din;                       % intrinsic diffusivity, um2/ms
Da = rms.Din;                       % axial diffusivity, um2/ms
za = 1.64;                          % z-score at alpha = 0.05
Nav = 64;                           % # gradient direction per b-shell
r_min_NP = zeros(numel(SNR), 1)-1;  % resolution limit of narrow-pulse
r_min_WP = zeros(numel(SNR), 1)-1;  % resolution limit of wide-pulse
for j = 1:numel(SNR)
    r_min_NP(j) = myelinADMrmin(max(bval), rms.del, rms.Del, D0, Da, za, SNR(j), Nav, 'narrow');
    r_min_WP(j) = myelinADMrmin(max(bval), rms.del, rms.Del, D0, Da, za, SNR(j), Nav, 'wide');
end

%% Plot figure, Canales-Rodriguez et al. 2005, narrow pulse solution
figure('unit','inch','position',[0 0 18 4]);
cmap = colormap('lines');
mk = {'o','v','s','^','d'};
clear h hx lgtxt
for i = 1:numel(n)
    subplot(1,4,i)
    hold on;
    for j = numel(SNR):-1:1
        plotstd(2*r_eff(:,i), 2*r_fit(:,j,1), 2*r_std(:,j,1), cmap(j,:), 0.3, 'area');
    end
end

for i = 1:numel(n)
    subplot(1,4,i)
    hold on;
    for j = 1:numel(SNR)
        if j==3
            h(j) = plot(r_eff(:,i)*2, r_fit(:,j,1)*2, mk{j}, 'color', cmap(j,:)*0.85,'linewidth',1);
        else
            h(j) = plot(r_eff(:,i)*2, r_fit(:,j,1)*2, mk{j}, 'color', cmap(j,:),'linewidth',1);
        end
        if ~isinf(SNR(j))
            if j==3
                hx = xline(r_min_NP(j)*2); set(hx, 'color', cmap(j,:)*0.85, 'linestyle', '--','linewidth',1);
            else
                hx = xline(r_min_NP(j)*2); set(hx, 'color', cmap(j,:), 'linestyle', '--','linewidth',1);
            end
        end
        lgtxt{j} = sprintf('SNR=%u', SNR(j));
    end
    xlim([0 4]);
    ylim([0 4]);
    yticks(0:4);
    hr = refline(1); set(hr, 'color', 'k');
    hl = plot(-1, -1, 'k--', 'linewidth', 1); lgtxt{numel(SNR)+1} = '2$r_{\rm min,NP}$';
    hg = legend([h hl], lgtxt, 'interpreter', 'latex', 'location', 'northwest', 'box', 'on', 'fontsize', 12);
    box on;
    pbaspect([1 1 1])
    switch n(i)
        case 0
            xlabel('2$\langle r\rangle$, $\mu$m', 'interpreter', 'latex', 'fontsize',20);
        case 1
            xlabel('2$r_{\rm vwa}$, $\mu$m', 'interpreter', 'latex', 'fontsize',20);
        case 2
            xlabel('2$r_{\rm eff, NP}$, $\mu$m', 'interpreter', 'latex', 'fontsize',20);
        case 4
            xlabel('2$r_{\rm eff, WP}$, $\mu$m', 'interpreter', 'latex', 'fontsize',20);
    end

    if i==1
        ylabel('fitted 2$r$, $\mu$m', 'interpreter', 'latex', 'fontsize',20);
    end
end

%% Plot figure, wide-pulse solution
figure('unit','inch','position',[0 0 18 4]);
cmap = colormap('lines');
mk = {'o','v','s','^','d'};
clear h hx lgtxt
for i = 1:numel(n)
    subplot(1,4,i)
    hold on;
    for j = numel(SNR):-1:1
        plotstd(2*r_eff(:,i), 2*r_fit(:,j,2), 2*r_std(:,j,2), cmap(j,:), 0.3, 'area');
    end
end

for i = 1:numel(n)
    subplot(1,4,i)
    hold on;
    for j = 1:numel(SNR)
        if j==3
            h(j) = plot(2*r_eff(:,i), 2*r_fit(:,j,2), mk{j}, 'color', cmap(j,:)*0.85,'linewidth',1);
        else
            h(j) = plot(2*r_eff(:,i), 2*r_fit(:,j,2), mk{j}, 'color', cmap(j,:),'linewidth',1);
        end
        if ~isinf(SNR(j))
            if j==3
                hx = xline(2*r_min_WP(j)); set(hx, 'color', cmap(j,:)*0.85, 'linestyle', '--','linewidth',1);
            else
                hx = xline(2*r_min_WP(j)); set(hx, 'color', cmap(j,:), 'linestyle', '--','linewidth',1);
            end
        end
        lgtxt{j} = sprintf('SNR=%u', SNR(j));
    end
    xlim([0 4]);
    ylim([0 4]);
    yticks(0:4);
    hr = refline(1); set(hr, 'color', 'k');
    hl = plot(-1, -1, 'k--', 'linewidth', 1); lgtxt{numel(SNR)+1} = '2$r_{\rm min,WP}$';
    if i==1
        hg = legend([h hl], lgtxt, 'interpreter', 'latex', 'location', 'southeast', 'box', 'on', 'fontsize', 12);
    else
        hg = legend([h hl], lgtxt, 'interpreter', 'latex', 'location', 'northwest', 'box', 'on', 'fontsize', 12);
    end
    box on;
    pbaspect([1 1 1])
    switch n(i)
        case 0
            xlabel('2$\langle r\rangle$, $\mu$m', 'interpreter', 'latex', 'fontsize',20);
        case 1
            xlabel('2$r_{\rm vwa}$, $\mu$m', 'interpreter', 'latex', 'fontsize',20);
        case 2
            xlabel('2$r_{\rm eff, NP}$, $\mu$m', 'interpreter', 'latex', 'fontsize',20);
        case 4
            xlabel('2$r_{\rm eff, WP}$, $\mu$m', 'interpreter', 'latex', 'fontsize',20);
    end
    if i==1
        ylabel('fitted 2$r$, $\mu$m', 'interpreter', 'latex', 'fontsize',20);
    end
end

%% Plot figure, Canales Rodriguez et al. 2005, narrow-pulse full solution
figure('unit','inch','position',[0 0 18 4]);
cmap = colormap('lines');
mk = {'o','v','s','^','d'};
clear h hx lgtxt
for i = 1:numel(n)
    subplot(1,4,i)
    hold on;
    for j = 1:numel(SNR)
        h(j) = plot(2*r_eff(:,i), 2*r_fit_CR(:,j), mk{j}, 'color', cmap(j,:),'linewidth',1);
        if ~isinf(SNR(j))
            if j==3
                hx = xline(2*r_min_NP(j)); set(hx, 'color', cmap(j,:)*0.85, 'linestyle', '--','linewidth',1);
            else
                hx = xline(2*r_min_NP(j)); set(hx, 'color', cmap(j,:), 'linestyle', '--','linewidth',1);
            end
        end
        lgtxt{j} = sprintf('SNR=%u', SNR(j));
    end
    xlim([0 4]);
    ylim([0 4]);
    hr = refline(1); set(hr, 'color', 'k');
    hl = plot(-1, -1, 'k--', 'linewidth', 1); lgtxt{numel(SNR)+1} = '2$r_{\rm min,NP}$';
    if i==1
        hg = legend([h hl], lgtxt, 'interpreter', 'latex', 'location', 'northwest', 'box', 'on', 'fontsize', 12);
    else
        hg = legend([h hl], lgtxt, 'interpreter', 'latex', 'location', 'northwest', 'box', 'on', 'fontsize', 12);
    end
    box on;
    pbaspect([1 1 1])
    switch n(i)
        case 0
            xlabel('2$\langle r\rangle$, $\mu$m', 'interpreter', 'latex', 'fontsize',20);
        case 1
            xlabel('2$r_{\rm vwa}$, $\mu$m', 'interpreter', 'latex', 'fontsize',20);
        case 2
            xlabel('2$r_{\rm eff, NP}$, $\mu$m', 'interpreter', 'latex', 'fontsize',20);
        case 4
            xlabel('2$r_{\rm eff, WP}$, $\mu$m', 'interpreter', 'latex', 'fontsize',20);
    end
    if i==1
        ylabel('fitted 2$r$, $\mu$m', 'interpreter', 'latex', 'fontsize',20);
    end
end

%% Plot histology information

% inner radius, um
r = 0.1:0.1:5;
r = r(:);

% thickness of myelin layer, um
lm = 12/1e3;

% # myelin layers
C0 = 0.35/lm;
C1 = 0.006/lm;
C2 = 0.024/lm;
Nm = round(C0 + C1*2*r + C2*log(2*r));

% Total myelin thickness, um
Lm = Nm*lm;

% g-ratio: ratio of inner to outer radii
gratio = r./(r+Lm);

figure('unit','inch','position',[0 0 15 5])

% plot inner diameter vs g-ratio
subplot(131)
hold on;
hp = plot(2*r, gratio, 'o', 'linewidth', 1);

xx = linspace(0, 10, 100);
yy = C0 + C1*xx + C2*log(xx);
yy = xx./(xx + 2*yy*lm);
hh = plot(xx, yy, '-', 'linewidth', 1);

legend([hh, hp], {'log-linear', 'numerical phantom'}, 'interpreter', 'latex', ...
    'box', 'off', 'location', 'southeast', 'fontsize', 14)

pbaspect([1 1 1]);
xlabel('inner diameter 2$r_i$, $\mu$m', 'interpreter', 'latex', 'fontsize', 20);
ylabel('$g$-ratio', 'interpreter', 'latex', 'fontsize', 20)
xlim([0 10]);
ylim([0 1]);
yticks(0:0.2:1);
box on;

% plot inner diameter vs # myelin layers
subplot(132)
hold on;
hp = plot(2*r, Nm, 'o', 'linewidth', 1);

xx = linspace(0, 10, 100);
yy = C0 + C1*xx + C2*log(xx);
hh =plot(xx, yy, '-', 'linewidth', 1);

legend([hh, hp], {'log-linear', 'numerical phantom'}, 'interpreter', 'latex', ...
    'box', 'off', 'location', 'southeast', 'fontsize', 14)

pbaspect([1 1 1]);
xlabel('inner diameter 2$r_i$, $\mu$m', 'interpreter', 'latex', 'fontsize', 20);
ylabel('\# myelin layers $N_m$', 'interpreter', 'latex', 'fontsize', 20)
xlim([0 10]); ylim([25 40])
box on;

% plot axon diameter distribution in Gamma distribution
di_mean = 2*(0.25:0.05:1).';
di_var  = (di_mean/2).^2;
b = di_var./di_mean;
a = di_mean./b;
di = 2*(0.1:0.1:5); di = di(:);
Ni = zeros(numel(di), numel(di_mean));
for j = 1:numel(di_mean)
    Ni(:,j) = gampdf(di, a(j), b(j));
end

ax = subplot(133);
hold on;
plot([2.5 2.5], [0.9 1.9], 'k-', 'linewidth', 1.5)
text(1.5, 1.4, '2$\bar{r}_i$', 'interpreter', 'latex', 'fontsize', 14);
cmap = colormap('parula');
list = 1:numel(di_mean);
Pi = Ni./sum(Ni,1)/mean(diff(di));
clear h lgtxt
for i = 1:numel(list)
    h(i) = plot(di, Pi(:,list(i)), '.-', 'color', cmap(i*floor(size(cmap,1)/numel(list)),:));
end
pbaspect([1 1 1]);
box on;
xlabel('inner diameter 2$r_i$, $\mu$m', 'interpreter', 'latex', 'fontsize', 20);
ylabel('PDF, $\mu$m$^{-1}$', 'interpreter', 'latex', 'fontsize', 20)
for i = 1:numel(list)
    lgtxt{i} = sprintf('%.2f $\\mu$m', di_mean(list(i)));
end
legend(h, lgtxt, 'interpreter','latex','fontsize',12, 'box', 'off', ...
    'NumColumns',2);
xlim([0 10]);
ylim([0 2]);
yticks(0:5)


