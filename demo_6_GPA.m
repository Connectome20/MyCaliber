% ********** Setup the directory on your computer **********
% demo 6: Compare analytical solutions for cylindrical shells of finite 
% thickness vs for cylindrical surface of 0 thickness, impermeable membrane
clear
restoredefaultpath
filePath = matlab.desktop.editor.getActiveFilename;
root0 = fileparts(filePath);
addpath(genpath(fullfile(root0,'lib')));

root = fullfile(root0,'data');

%% Compare solutions of cylindrical shells vs cylindrical surface

% inner radius, um
r = [1 2 3];

% thickness-to-inner radius ratio
lrR = [1/100, 1/10, 1/5, 1/2];

% intrinsic diffusivity, um2/ms
D0 = 0.8;

% narrow pulse or wide pulse
pulsetype = {'narrow','wide'};

% pulse width, ms
delta_wide = [0.1:0.1:0.9 1 2:2:50];

% width of refocusing RF pulse, ms
tRF180 = 5;

% diffusion time, ms
Delta_wide = delta_wide+tRF180;

% RD solution of cylindrical surface of zero thickness
% narrow-pulse solution: Canales-Rodriguez et al. 2005
% wide-pulse solution: GPA solution
RD_surface = zeros(numel(lrR), numel(r), numel(pulsetype), numel(Delta_wide));

% RD solution of cylindrical shells of finite thickness: GPA solution
% Lebois, A. (2014). Brain microstructure mapping using quantitative and 
% diffsusion MRI (Doctoral dissertation, Université Paris Sud-Paris XI).
RD_shell = zeros(numel(lrR), numel(r), numel(Delta_wide));
for i = 1:numel(lrR)
    lrRi = lrR(i);
    for j = 1:numel(r)
        li = r(j)*lrRi;     % myelin layer thickness, um
        ri = r(j);          % inner radius, um
        ro = r(j) + li;     % outer radius, um
        rm = (ro+ri)/2;     % cylindrical shell radius at the middle thickness, um

        for k = 1:2
            RD_surface(i,j,k,:) = myelinRDmodel(rm, delta_wide, Delta_wide, D0, pulsetype{k});
        end
        RD_shell(i,j,:) = coCylRDmodel(ri, ro, delta_wide, Delta_wide, D0);
    end
end

figure('unit','inch','position',[0 0 15 4]);
cmap = colormap('lines');
for i = 1:numel(lrR)

    subplot(1, numel(lrR)+1, i);
    hold on;
    for j = 1:numel(r)
        RD_surface_NP = squeeze(RD_surface(i,j,1,:));
        RD_surface_WP = squeeze(RD_surface(i,j,2,:));
        RD_shell_j = squeeze(RD_shell(i,j,:));
        plot(Delta_wide, RD_surface_NP, '--', 'color', cmap(j,:), 'linewidth', 1);
        plot(Delta_wide, RD_surface_WP, '-', 'color', cmap(j,:), 'linewidth', 1);
        plot(Delta_wide(10:end), RD_shell_j(10:end), '.', 'color', cmap(j,:), 'markersize', 8);
    end
    
    xlim([0 50]); xticks(0:10:50);
    ylim([0 0.3]); yticks(0:0.05:0.3);
    
    pbaspect([1 1 1]);
    box on; grid on;
    xlabel('$\Delta$, ms','Interpreter','latex','FontSize',16);
    if i==1, ylabel('$D^\perp(\Delta,\delta)$, $\mu$m$^2$/ms','Interpreter','latex','FontSize',16); end
    title(sprintf('$l_m/r_i=$ %u\\%%',lrR(i)*100),'interpreter','latex','fontsize',16);
end


subplot(1, numel(lrR)+1, (numel(lrR)+1)*1);
hold on;
clear h lgtxt
for j = 1:numel(r)
    h(j) = plot(-1, -1, '.-', 'color', cmap(j,:), 'linewidth', 1, 'markersize', 6);
    lgtxt{j} = sprintf('$2r_i=$ %.1g $\\mu$m', 2*r(j));
end
h(j+1) = plot(-1, -1, 'k.' , 'markersize', 10);
h(j+3) = plot(-1, -1, 'k--', 'linewidth', 1);
h(j+2) = plot(-1, -1, 'k-' , 'linewidth', 1);
lgtxt{j+1} = 'finite $l_m$ solution';
lgtxt{j+3} = 'Canales-Rodr\''iguez et al. (2)';
lgtxt{j+2} = 'proposed solution (6)';
lg = legend(h, lgtxt, 'box', 'off', 'Interpreter', 'latex', 'location', 'west', 'fontsize', 12);
lg.Position = lg.Position + [-0.03 0 0 0];
xlim([0 1]); ylim([0 1]);
pbaspect([1 1 1]);
axis off


