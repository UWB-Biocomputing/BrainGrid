function [fig1, fig2, fig3, fig4, fig5] = growth2()

global stateoutfile;
global ratesHistory;
global radiiHistory;
global burstinessHist;
global xloc;
global yloc;
global neuronTypes;
global starterNeurons;    % 1 based indexing
global now;    % read now from history dump
global Tsim;                % read Tsim from history dump
global numNeurons;
global xlen;
global ylen;
global INH;
global EXC;

fprintf('Plotting raster.\n');
% Plot activity during the final simulation segment
fig1 = figure(1); clf;
plot_channels(ratesHistory, Tsim, now);
xlabel('Time (s)')
ylabel('Neuron Number')
title(['Raster Plot: ', [stateoutfile '.xml']])

fprintf('Plotting radius and firing rate changes.\n');
% Plot radius growth history
% INH neuron - RED ([1 0 0])size(
% Starter neuron - BLUE ([0 1 0])
% Edge neuron (not INH or starter) - GREEN ([0 0 1])
% Others - BLACK ([0 0 0])
fig2 = figure(2); clf;
subplot(2, 1, 1);
defColorOrder = get(0, 'DefaultAxesColorOrder');
colorOrder = zeros(length(neuronTypes), 3);     % set all BLACK
xedge = union(find(xloc == 0), find(xloc == xlen-1));
yedge = union(find(yloc == 0), find(yloc == ylen-1));
corner = intersect(xedge, yedge);

colorOrder(setdiff(setdiff(union(xedge, yedge), starterNeurons), corner), 1) = 0;         % set GREEN (edge neurons)
colorOrder(setdiff(setdiff(union(xedge, yedge), starterNeurons), corner), 2) = 1;
colorOrder(setdiff(setdiff(union(xedge, yedge), starterNeurons), corner), 3) = 0;
colorOrder(corner, 1) = 0;              % set CYAN (corner neurons)
colorOrder(corner, 2) = 1;
colorOrder(corner, 3) = 1;
colorOrder(find(neuronTypes == INH), 1) = 1;    % set RED (INH neurons)
colorOrder(find(neuronTypes == INH), 2) = 0;
colorOrder(find(neuronTypes == INH), 3) = 0;
colorOrder(starterNeurons, 1) = 0;              % set BLUE (starter neurons)
colorOrder(starterNeurons, 2) = 0;
colorOrder(starterNeurons, 3) = 1;
% EXC that are not starters are BLACK                                             
set(0, 'DefaultAxesColorOrder', colorOrder);
set(gca,'FontSize',18,'LineWidth',2)
plot([0:Tsim:now], radiiHistory');
xlabel('Time (s)');
ylabel('Radii');
%title(['Radius Plot: ', [stateoutfile '.xml']])

% fprintf('Plotting increase rate of radius.\n')
% % Plot slope of radius growth history
% radiiSlope = diff(radiiHistory);
% subplot(2, 1, 1);
% plot([0:Tsim:now-Tsim], radiiSlope);
% xlabel('Time (s)');
% ylabel('Increase rate of radii');
% title(['Radius Increase rate Plot: ', [stateoutfile '.xml']])

fprintf('Plotting ratime.\n');
% and firing rate versus time, in bins that are Tsim wide
subplot(2, 1, 2);
plot([0:Tsim:now], ratesHistory');
set(gca,'FontSize',18,'LineWidth',2)
xlabel('Time (s)');
ylabel('Firing rate (Hz)');
title(['Ratime Plot: ', [stateoutfile '.xml']])

% And final radii
fig3 = figure(3); clf;
plotradii(xloc, yloc, radiiHistory, neuronTypes, starterNeurons)
title(['Final Radii', stateoutfile])

% and final firing rates
fig4 = figure(4); clf;
plotrates(xloc, yloc, ratesHistory, neuronTypes, starterNeurons)
title(['Final firing rates: ', [stateoutfile '.xml']])

set(0, 'DefaultAxesColorOrder', defColorOrder);

fprintf('Plotting burstiness evolution');
% and burstiness index
fig5 = figure(5); clf;
burstiness(burstinessHist);
%title(['Burstiness evolution: ', [stateoutfile '.xml']])



