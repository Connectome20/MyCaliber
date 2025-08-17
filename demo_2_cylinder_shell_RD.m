% ********** Setup the directory on your computer **********
% demo 2: MC simulations of RD in coaxial cylindrical shell of finite
% thickness, 1 layer, impermeable membrane
clear
restoredefaultpath
filePath = matlab.desktop.editor.getActiveFilename;
root0 = fileparts(filePath);
addpath(genpath(fullfile(root0,'lib')));

root = fullfile(root0,'data');
root_cuda = fullfile(root0,'lib','rms');

% project name
projname = 'coaxial_cyliner_shell_RD';

mkdir(fullfile(root,projname));

%% Generate packing

% inner radius, um
r = [1, 2, 3];

% thickness of each myelin layer, um
% myelin layer thickness lm should be larger than the step size sqrt(4*D0*dt)
lm = 12/1e3;

% number of layers
Nm = [1, 1, 1];

% pulse width, ms
Td = 6:10;

% width of refocusing RF pulse, ms
tRF180 = 5;

% diffusion time, ms
TD = Td+tRF180;

% b-value, ms/um2
bval = 1:6;

% gradient direction perpendicular to cylindrical axis (z-axis)
theta = linspace(0, pi, 5); theta = theta(1:end-1); theta = theta(:);
bvec = [cos(theta), sin(theta), 0*theta];

% simulation parameters
dt = 1e-6;                      % time of each step, ms
TN = ceil(max(TD+Td)/dt)+100;   % # steps
NPar = 1e5;                     % # random walkers
D0 = 0.8;                       % intrinsic diffusivity, um2/ms
threadpb = 256;                 % thread per block for CUDA

seed = 0;
for j = 1:numel(r)
    seed = seed + 1;
    target = fullfile(root,projname,sprintf('CoCyl_%04u',seed));
    mkdir(target);

    % field of view, um
    res = 2*r(j)*1.05;

    % save simulation parameters
    fileID = fopen(fullfile(target,'simParamInput.txt'),'w');
    fprintf(fileID,sprintf('%g\n%u\n%u\n%g\n%u\n%g\n%g\n%u\n%g\n',...
        dt, TN, NPar, D0, threadpb, ...
        r(j)/res, lm/res, Nm(j), res));
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
for i = 1:3
    target = fullfile(root,projname,sprintf('CoCyl_%04u',i));
    fprintf(fileID,sprintf('cd %s\n',target));
    fprintf(fileID,sprintf('cp -a %s .\n',fullfile(root_cuda,'main_PGSE_cuda')));
    fprintf(fileID,'./main_PGSE_cuda\n');
end
fclose(fileID);

% Open the terminal window in the project folder and run "sh job.sh"

% You may need to open the terminal in the root_cuda folder and compile the 
% CUDA code using "nvcc main_PGSE.cu -o main_PGSE_cuda"

%% Plot simulation results
figure('unit','inch','position',[0 0 10 5]);
clear h ht lgtxt
cmap = colormap('lines');
mk = {'o','v','s'};

% Radial diffusivity of narrow-pulse sequence
subplot(121)
hold on;
for j = 1:3
    rms = simul3DcoCyl_cuda_pgse_bvec(fullfile(root,projname,sprintf('CoCyl_%04u',j)));
    theta = linspace(0, pi, 5); theta = theta(1:end-1); theta = theta(:);
    bvec = [cos(theta), sin(theta), 0*theta];
    [~, RD] = rms.akc_mom(bvec);
    RD = mean(RD, 2);
    
    % cylindrical shell radius at the middle thickness, um
    r = rms.rCir + rms.Nm.*rms.lm - rms.lm/2;
    D0 = rms.Din;               % intrinsic diffusivity, um2/ms
    tm = r^2/D0;                % correlation time, ms

    t_plot = 0.01:0.01:100;     % diffusion time for plotting, ms

    % Theory
    Dm = @(t) r^2/2./t.*(1-exp(-t/tm));
    D_plot = Dm(t_plot);
    
    plot(1./rms.TD(1:20:end),RD(1:20:end),mk{j},'markersize',6,'linewidth',1, 'color', cmap(j,:)); 
    plot(1./t_plot, D_plot, '-k', 'linewidth', 1, 'color', cmap(j,:));

    h(j) = plot(-1, -1,mk{j},'markersize',6,'linewidth',1,'Color',cmap(j,:));
    lgtxt{j} = sprintf('2$r_i=$%u $\\mu$m',round(r)*2);
end
xlim([0 0.5]); ylim([0 0.5])
box on; grid on;
set(gca,'fontsize',12);
pbaspect([1 1 1]);
ht = plot(-1, -1, 'k-', 'linewidth',1);
lgtxt{4} = 'theory';
legend([h ht],lgtxt,'Interpreter','latex','fontsize',14, ...
    'location','northwest','box','off');
xlabel('$1/t$, ms$^{-1}$','Interpreter','latex','FontSize',20);
ylabel('$D^\perp(t)$, $\mu$m$^2$/ms','Interpreter','latex','FontSize',20);
title('narrow pulse','interpreter','latex','fontsize',20);

% Radial diffusivity of wide-pulse sequence
subplot(122);
r_fit = zeros(3,1);
r_mom = zeros(3,1);
n_dir = 4;
for j = 1:3
    rms = simul3DcoCyl_cuda_pgse_bvec(fullfile(root,projname,sprintf('CoCyl_%04u',j)));
    sig = rms.sig;
    bval = unique(rms.bval);
    A = [-bval, 1/6*bval.^2, bval.^(3:4)];
    adc = zeros(rms.NDel,1);
    for i = 1:rms.NDel
        sigi = sig(:,i);
        sigi = reshape(sigi, n_dir, []);
        sigi = mean(sigi,1);
        sigi = sigi(:);
        list = 1:6;
        X = A(list,:)\log(sigi(list));
        adc(i) = X(1);
    end
    
    % cylindrical shell radius at the middle thickness, um
    rm = rms.rCir + (1:rms.Nm)*rms.lm - rms.lm/2;
    D0 = rms.Din;               % intrinsic diffusivity, um2/ms
    delta = rms.del;            % pulse width, ms
    Delta = rms.Del;            % diffusion time, ms

    hold on;
    h(j) = plot(1./delta./(Delta-delta/3), adc, mk{j},...
        'color', cmap(j,:), 'markersize', 6, 'linewidth', 1);
    
    % width of refocusing RF pulse, ms
    tRF180 = 5;

    % pulse width for plotting, ms
    delta_plot = 1:0.1:100;
    
    % diffusion time for plotting, ms
    Delta_plot = delta_plot+tRF180;
    
    % theory
    D_plot = myelinRDmodel(rm, delta_plot, Delta_plot, D0, 'wide');
    
    % approximate solution from Canales-Rodriguez et al. 2025
    D_plot_CR = myelinRDmodel(rm, delta_plot, Delta_plot, D0, 'narrow');
    
    plot(1./delta_plot./(Delta_plot-delta_plot/3), D_plot, ...
        'color', cmap(j,:), 'linewidth', 1);
    plot(1./delta_plot./(Delta_plot-delta_plot/3), D_plot_CR, '--', ...
        'color', cmap(j,:), 'linewidth', 1);
    lgtxt{j} = sprintf('2$r_i=$%u $\\mu$m',round(rms.rCir)*2);
end
ht = plot(-1, -1, 'k-', 'linewidth',1);
ht_CR = plot(-1, -1, 'k--', 'linewidth', 1);
lgtxt{4} = 'theory';
lgtxt{5} = 'Canales-Rodr\''iguez et al.';
xlim([0 0.02]); ylim([0 0.25])
box on; grid on;
set(gca,'fontsize',12);
pbaspect([1 1 1]);
hg = legend([h ht ht_CR],lgtxt,'Interpreter','latex','fontsize',14,...
    'location','northwest','box','off');
xlabel('$1/[\delta\cdot(\Delta-\delta/3)]$, ms$^{-2}$','Interpreter','latex','FontSize',20);
ylabel('$D^\perp(\Delta,\delta)$, $\mu$m$^2$/ms','Interpreter','latex','FontSize',20);
title('wide pulse','interpreter','latex','fontsize',20);


