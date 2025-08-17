% ********** Setup the directory on your computer **********
% demo 1: Simulation on a cylindrical surface of infinitely thin layer
clear
restoredefaultpath
filePath = matlab.desktop.editor.getActiveFilename;
root0 = fileparts(filePath);
addpath(genpath(fullfile(root0,'lib')));

%% Narrow pulse and wide pulse simulation, radial diffusivity
delta = single(6:10);           % pulse width, ms
tRF180 = 5;                     % width of refocusing RF pulse, ms
Delta = single(delta + tRF180); % diffusion time, ms
dt = single(1e-3);              % time step, ms
Nt = ceil(max(Delta+delta)/dt); % # steps
Np = single(1e5);               % # random walkers
rs = single([1 2 3]);           % cylinder radius, um
D0 = single(0.8);               % intrinsic diffusivity, um2/ms
bval = single(1:6);             % b-value, ms/um2

% gradient direction, perpendicular to cylinder axis (z-axis)
theta = linspace(0, pi, 5); theta = theta(1:end-1); theta = theta(:);
bvec = [cos(theta), sin(theta), 0*theta];
bvec = single(bvec);

% Apparent diffusivity of narrow-pulse sequence, um2/ms
Dnp = zeros(numel(rs), Nt, 'single');

% Diffusion signal of wide-pulse sequence
Swp = zeros(numel(rs), numel(Delta), numel(bval), size(bvec, 1), 'single');

% Monte Carlo simulations
% tnp: diffusion time for narrow-pulse sequence, ms
tic;
for i = 1:numel(rs)
    [tnp, Dnp(i,:), Swp(i,:,:,:)] = cylinderSurfaceSimulation(rs(i), delta, Delta, bval, bvec, D0, dt, Np);
end
toc;

% Calculate apparent radial diffusivity of wide-pulse sequence, um2/ms
sig = mean(Swp, 4);
A = [-bval(:) bval(:).^(2:4)];
sig = reshape(sig, [], numel(bval));
X = A\log(abs(sig).'+eps);
Dwp = X(1,:).';
Dwp = reshape(Dwp, numel(rs), numel(Delta));

%% Plot figure, radial diffusivity
close all
figure('unit','inch','position',[0 0 10 5]);
cmap = colormap('lines');
mk = {'o','v','s'};

% Radial diffusivity of narrow-pulse sequence
clear h ht lgtxt
subplot(121);
hold on;
for i = 1:numel(rs)
    r = rs(i);
    tm = r^2/D0; % Correlation time, ms

    % diffusion time for plotting, ms
    t_plot = 1:0.1:100;

    % Theory
    Dm = @(t) r^2/2./t.*(1-exp(-t/tm));
    D_plot = Dm(t_plot);
    
    plot(1./tnp(1:500:end),Dnp(i,1:500:end),mk{i},...
        'markersize',6,'linewidth',1, 'color', cmap(i,:)); 
    plot(1./t_plot, D_plot, '-k', 'linewidth', 1, 'color', cmap(i,:));
    xlim([0 0.5]); ylim([0 0.5])
    box on; grid on;
    set(gca,'fontsize',12);
    pbaspect([1 1 1]);
    h(i) = plot(-1, -1,mk{i},'markersize',6,'linewidth',1,'Color',cmap(i,:));
    lgtxt{i} = sprintf('2$r=$%u $\\mu$m',r*2);
end
ht = plot(-1, -1, 'k-', 'linewidth',1);
lgtxt{4} = 'theory';
legend([h ht],lgtxt,'Interpreter','latex','fontsize',14, ...
    'location','northwest','box','off');
xlabel('$1/t$, ms$^{-1}$','Interpreter','latex','FontSize',20);
ylabel('$D^\perp(t)$, $\mu$m$^2$/ms','Interpreter','latex','FontSize',20);
title('narrow pulse','interpreter','latex','fontsize',20);

% Radial diffusivity of wide-pulse sequence
clear h ht lgtxt
subplot(122);
hold on;
for i = 1:numel(rs)
    r = rs(i);

    % pulse width for plotting, ms
    delta_plot = 1:0.1:100;

    % diffusion time for plotting, ms
    Delta_plot = delta_plot + tRF180;

    % Theory
    D_plot = myelinRDmodel(r, delta_plot, Delta_plot, D0, 'wide');

    % Approximate solution from Canales-Rodriguez et al. 2025
    D_plot_CR = myelinRDmodel(r, delta_plot, Delta_plot, D0, 'narrow');

    plot(1./delta./(Delta-delta/3), Dwp(i,:), mk{i}, ...
        'markersize', 6, 'linewidth', 1, 'color', cmap(i,:));
    plot(1./delta_plot./(Delta_plot-delta_plot/3), D_plot, '-', ...
        'linewidth', 1, 'color', cmap(i,:));
    plot(1./delta_plot./(Delta_plot-delta_plot/3), D_plot_CR, '--', ...
        'linewidth', 1, 'color', cmap(i,:));
    h(i) = plot(-1, -1 ,mk{i}, 'markersize', 6,'linewidth', 1, 'color', cmap(i,:));
    lgtxt{i} = sprintf('2$r=$%u $\\mu$m', r*2);
end
ht = plot(-1, -1, 'k-', 'linewidth',1);
lgtxt{4} = 'theory';
ht_CR = plot(-1, -1, 'k--', 'linewidth', 1);
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
