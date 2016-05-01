function [fig2a, fig2b, fig2c, fig2d, fig2e, fig2f, fig2g] = growth2a()

global stateoutfile;
global ratesHistory;
global radiiHistory;
global burstinessHist;
global xloc;
global yloc;
global neuronTypes;
global starterNeurons;      % 1 based indexing
global now;                 % read now from history dump
global Tsim;                % length of an epoch
global numSims;             % number of epochs
global numNeurons;
global xlen;
global ylen;
global INH;
global EXC;

fprintf('Plotting radius and firing rate changes.\n');
% Plot radius growth history
% INH neuron - RED ([1 0 0])
% Starter neuron - BLUE ([0 1 0])
% Edge neuron (not INH or starter) - GREEN ([0 0 1])
% Edge & starter - ([0 1 1])
% Others - BLACK ([0 0 0])
colorOrder = zeros(length(neuronTypes), 3);     % set all BLACK
xedge = union(find(xloc == 0), find(xloc == xlen-1));
yedge = union(find(yloc == 0), find(yloc == ylen-1));
corner = intersect(xedge, yedge);

ratesHistory_edge  = ratesHistory(:, setdiff(setdiff(union(xedge, yedge), starterNeurons), corner));
ratesHistory_cnr   = ratesHistory(:, corner);
ratesHistory_inh   = ratesHistory(:, find(neuronTypes == INH));
ratesHistory_st    = ratesHistory(:, starterNeurons);
ratesHistory_other = ratesHistory(:, setdiff(1:numNeurons, union(starterNeurons, union(union(xedge, yedge), find(neuronTypes == INH)))));

radiiHistory_edge  = radiiHistory(:, setdiff(setdiff(union(xedge, yedge), starterNeurons), corner));
radiiHistory_cnr   = radiiHistory(:, corner);
radiiHistory_inh   = radiiHistory(:, find(neuronTypes == INH));
radiiHistory_st    = radiiHistory(:, starterNeurons);
radiiHistory_other = radiiHistory(:, setdiff(1:numNeurons, union(starterNeurons, union(union(xedge, yedge), find(neuronTypes == INH)))));

colorOrder_g = colorOrder;
colorOrder_r = colorOrder;
colorOrder_b = colorOrder;
colorOrder_k = colorOrder;
% colorOrder_g(:, 1) = 0;         % set GREEN (edge neurons)
% colorOrder_g(:, 2) = 1;
% colorOrder_g(:, 3) = 0;
% colorOrder_c(:, 1) = 0;         % set CYAN (corner neurons)
% colorOrder_c(:, 2) = 1;
% colorOrder_c(:, 3) = 1;
% colorOrder_r(:, 1) = 1;         % set RED (INH neurons)
% colorOrder_r(:, 2) = 0;
% colorOrder_r(:, 3) = 0;
% colorOrder_b(:, 1) = 0;         % set BLUE (starter neurons)
% colorOrder_b(:, 2) = 0;
% colorOrder_b(:, 3) = 1;
% colorOrder_k(:, 1) = 0;         % set BLACK (other neurons)
% colorOrder_k(:, 2) = 0;
% colorOrder_k(:, 3) = 0;
% set all BLACK
colorOrder_g(:, 1) = 0;         % set GREEN (edge neurons)
colorOrder_g(:, 2) = 0;
colorOrder_g(:, 3) = 0;
colorOrder_c(:, 1) = 0;         % set CYAN (corner neurons)
colorOrder_c(:, 2) = 0;
colorOrder_c(:, 3) = 0;
colorOrder_r(:, 1) = 0;         % set RED (INH neurons)
colorOrder_r(:, 2) = 0;
colorOrder_r(:, 3) = 0;
colorOrder_b(:, 1) = 0;         % set BLUE (starter neurons)
colorOrder_b(:, 2) = 0;
colorOrder_b(:, 3) = 0;
colorOrder_k(:, 1) = 0;         % set BLACK (other neurons)
colorOrder_k(:, 2) = 0;
colorOrder_k(:, 3) = 0;
ylim_r_max=7;   % maximum radius
ylim_f_max=6;   % maximum firing rate
max(ratesHistory_other(:,1))
if max(ratesHistory_other(:,1)) > ylim_f_max
    ylim_f_max=400;
end

%%%%%%%%%%%%%% draw edge neurons 
fig2a = figure(6); clf;
subplot(2, 1, 1);
% subplot(3, 1, 1);
defColorOrder = get(0, 'DefaultAxesColorOrder');
set(0, 'DefaultAxesColorOrder', colorOrder_g);
plot([0:Tsim:now], radiiHistory_edge');
set(gca,'FontSize',18,'LineWidth',2)
xlabel('Time (s)');
ylabel('Radii');
xlim([0 60000])
ylim([0 ylim_r_max])
%title(['Radius Plot: ', [stateoutfile '.xml']])
%title('Radius Plot')

% fprintf('Plotting increase rate of radius.\n')
% % Plot slope of radius growth history
% radiiSlope_edge = diff(radiiHistory_edge);
% subplot(3, 1, 2);
% plot([0:Tsim:now-Tsim], radiiSlope_edge);
% set(gca,'FontSize',18,'LineWidth',2)
% xlabel('Time (s)');
% ylabel('Increase rate of radii');
% ylim([-0.01 0.01])
%title(['Radius Increase rate Plot: ', [stateoutfile '.xml']])
%title('Radius Increase rate Plot')

fprintf('Plotting ratime.\n');
% and firing rate versus time, in bins that are Tsim wide
subplot(2, 1, 2);
% subplot(3, 1, 3);
plot([0:Tsim:now], ratesHistory_edge');
set(gca,'FontSize',18,'LineWidth',2)
xlabel('Time (s)');
ylabel('Firing rate (Hz)');
xlim([0 60000])
ylim([0 ylim_f_max])   
%title(['Ratime Plot: ', [stateoutfile '.xml']])
%title('Ratime Plot')

%%%%%%%%%%%%%% draw corner neurons
fig2b = figure(7); clf;
% subplot(3, 1, 1);
subplot(2, 1, 1);
set(0, 'DefaultAxesColorOrder', colorOrder_c);
plot([0:Tsim:now], radiiHistory_cnr');
set(gca,'FontSize',18,'LineWidth',2)
xlabel('Time (s)');
ylabel('Radii');
xlim([0 60000])
ylim([0 ylim_r_max])
%title(['Radius Plot: ', [stateoutfile '.xml']])
%title('Radius Plot')

% fprintf('Plotting increase rate of radius.\n')
% % Plot slope of radius growth history
% radiiSlope_cnr = diff(radiiHistory_cnr);
% subplot(3, 1, 2);
% plot([0:Tsim:now-Tsim], radiiSlope_cnr);
% set(gca,'FontSize',18,'LineWidth',2)
% xlabel('Time (s)');
% ylabel('Increase rate of radii');
% ylim([-0.01 0.01])
%title(['Radius Increase rate Plot: ', [stateoutfile '.xml']])
%title('Radius Increase rate Plot')

fprintf('Plotting ratime.\n');
% and firing rate versus time, in bins that are Tsim wide
% subplot(3, 1, 3);
subplot(2, 1, 2);
plot([0:Tsim:now], ratesHistory_cnr');
set(gca,'FontSize',18,'LineWidth',2)
xlabel('Time (s)');
ylabel('Firing rate (Hz)');
xlim([0 60000])
ylim([0 ylim_f_max])  
%title(['Ratime Plot: ', [stateoutfile '.xml']])
%title('Ratime Plot')

%%%%%%%%%%%%%% draw INH neurons
fig2c = figure(8); clf;
% subplot(3, 1, 1);
subplot(2, 1, 1);
set(0, 'DefaultAxesColorOrder', colorOrder_r);
plot([0:Tsim:now], radiiHistory_inh');
set(gca,'FontSize',18,'LineWidth',2)
xlabel('Time (s)');
ylabel('Radii');
xlim([0 60000])
ylim([0 ylim_r_max])
%title(['Radius Plot: ', [stateoutfile '.xml']])
%title('Radius Plot')

% fprintf('Plotting increase rate of radius.\n')
% % Plot slope of radius growth history
% radiiSlope_inh = diff(radiiHistory_inh);
% subplot(3, 1, 2);
% plot([0:Tsim:now-Tsim], radiiSlope_inh);
% set(gca,'FontSize',18,'LineWidth',2)
% xlabel('Time (s)');
% ylabel('Increase rate of radii');
% ylim([-0.01 0.01])
%title(['Radius Increase rate Plot: ', [stateoutfile '.xml']])
%title('Radius Increase rate Plot')

fprintf('Plotting ratime.\n');
% and firing rate versus time, in bins that are Tsim wide
% subplot(3, 1, 3);
subplot(2, 1, 2);
plot([0:Tsim:now], ratesHistory_inh');
set(gca,'FontSize',18,'LineWidth',2)
xlabel('Time (s)');
ylabel('Firing rate (Hz)');
xlim([0 60000])
ylim([0 ylim_f_max])   
%title(['Ratime Plot: ', [stateoutfile '.xml']])
%title('Ratime Plot')

%%%%%%%%%%%%%% draw starter neurons
fig2d = figure(9); clf;
% subplot(3, 1, 1);
subplot(2, 1, 1);
set(0, 'DefaultAxesColorOrder', colorOrder_b);
plot([0:Tsim:now], radiiHistory_st');
set(gca,'FontSize',18,'LineWidth',2)
xlabel('Time (s)');
ylabel('Radii');
xlim([0 60000])
ylim([0 ylim_r_max])
%title(['Radius Plot: ', [stateoutfile '.xml']])
%title('Radius Plot')

% fprintf('Plotting increase rate of radius.\n')
% % Plot slope of radius growth history
% radiiSlope_st = diff(radiiHistory_st);
% subplot(3, 1, 2);
% plot([0:Tsim:now-Tsim], radiiSlope_st);
% set(gca,'FontSize',18,'LineWidth',2)
% xlabel('Time (s)');
% ylabel('Increase rate of radii');
% ylim([-0.01 0.01])
%title(['Radius Increase rate Plot: ', [stateoutfile '.xml']])
%title('Radius Increase rate Plot')

fprintf('Plotting ratime.\n');
% and firing rate versus time, in bins that are Tsim wide
% subplot(3, 1, 3);
subplot(2, 1, 2);
plot([0:Tsim:now], ratesHistory_st');
set(gca,'FontSize',18,'LineWidth',2)
xlabel('Time (s)');
ylabel('Firing rate (Hz)');
xlim([0 60000])
ylim([0 ylim_f_max])   
%title(['Ratime Plot: ', [stateoutfile '.xml']])
%title('Ratime Plot')


%%%%%%%%%%%%%% draw other neurons
fig2e = figure(10); clf;
% subplot(3, 1, 1);
subplot(2, 1, 1);
set(0, 'DefaultAxesColorOrder', colorOrder_k);
plot([0:Tsim:now], radiiHistory_other');
set(gca,'FontSize',18,'LineWidth',2)
xlabel('Time (s)');
ylabel('Radii');
xlim([0 60000])
ylim([0 ylim_r_max])
%title(['Radius Plot: ', [stateoutfile '.xml']])
%title('Radius Plot')


% fprintf('Plotting increase rate of radius.\n')
% % Plot slope of radius growth history
% radiiSlope_other = diff(radiiHistory_other);
% subplot(3, 1, 2);
% plot([0:Tsim:now-Tsim], radiiSlope_other);
% set(gca,'FontSize',18,'LineWidth',2)
% xlabel('Time (s)');
% ylabel('Increase rate of radii');
% ylim([-0.01 0.01])
%title(['Radius Increase rate Plot: ', [stateoutfile '.xml']])
%title('Radius Increase rate Plot')

fprintf('Plotting ratime.\n');
% and firing rate versus time, in bins that are Tsim wide
% subplot(3, 1, 3);
subplot(2, 1, 2);
plot([0:Tsim:now], ratesHistory_other');
set(gca,'FontSize',18,'LineWidth',2)
xlabel('Time (s)');
ylabel('Firing rate (Hz)');
xlim([0 60000])
ylim([0 ylim_f_max])   
%title('Ratime Plot')
%title(['Ratime Plot: ', [stateoutfile '.xml']])


set(0, 'DefaultAxesColorOrder', defColorOrder);

%%%%%%%%%%%%%% draw distribution histogram of neurite radii (OTHERS) 
fig2f = figure(11);
fprintf('Final neurite radii: mean, std, and randge.\n');
mean(radiiHistory_other(numSims,:))
std(radiiHistory_other(numSims,:))
range(radiiHistory_other(numSims,:))
fprintf('Final neurite radii (inhibitory): mean, std, and randge.\n');
mean(radiiHistory_inh(numSims,:))
std(radiiHistory_inh(numSims,:))
range(radiiHistory_inh(numSims,:))
bin_size_hist=0.1;
n_bin=(range(radiiHistory_other(numSims,:)))/bin_size_hist
[n,xout]=hist(radiiHistory_other(numSims,:),n_bin);
bar(xout,n/length(radiiHistory_other(numSims,:)),1);
set(gca,'FontSize',18,'LineWidth',2)
xlabel('Radii')
ylabel('Probability')
xlim([1.6 3.6])
ylim([0 0.55])

% 
% mean and std plot (other neurns and inh neurons)
fig2g=figure(12);
hold on
mean_other(1)=0;
mean_other(2)=mean(radiiHistory_other(100,:));
mean_other(3)=mean(radiiHistory_other(200,:));
mean_other(4)=mean(radiiHistory_other(300,:));
mean_other(5)=mean(radiiHistory_other(400,:));
mean_other(6)=mean(radiiHistory_other(500,:));
mean_other(7)=mean(radiiHistory_other(600,:));
std_other(1)=0;
std_other(2)=std(radiiHistory_other(100,:));
std_other(3)=std(radiiHistory_other(200,:));
std_other(4)=std(radiiHistory_other(300,:));
std_other(5)=std(radiiHistory_other(400,:));
std_other(6)=std(radiiHistory_other(500,:));
std_other(7)=std(radiiHistory_other(600,:));
errorbar(mean_other,std_other/2)

median_other(1)=0;
median_other(2)=median(radiiHistory_other(100,:));
median_other(3)=median(radiiHistory_other(200,:));
median_other(4)=median(radiiHistory_other(300,:));
median_other(5)=median(radiiHistory_other(400,:));
median_other(6)=median(radiiHistory_other(500,:));
median_other(7)=median(radiiHistory_other(600,:));
% p=plot(median_other);
% set(p,'Color','red')

mean_inh(1)=0;
mean_inh(2)=mean(radiiHistory_inh(100,:));
mean_inh(3)=mean(radiiHistory_inh(200,:));
mean_inh(4)=mean(radiiHistory_inh(300,:));
mean_inh(5)=mean(radiiHistory_inh(400,:));
mean_inh(6)=mean(radiiHistory_inh(500,:));
mean_inh(7)=mean(radiiHistory_inh(600,:));
std_inh(1)=0;
std_inh(2)=std(radiiHistory_inh(100,:));
std_inh(3)=std(radiiHistory_inh(200,:));
std_inh(4)=std(radiiHistory_inh(300,:));
std_inh(5)=std(radiiHistory_inh(400,:));
std_inh(6)=std(radiiHistory_inh(500,:));
std_inh(7)=std(radiiHistory_inh(600,:));
p=errorbar(mean_inh,std_inh/2)
%set(p,'Color','red')
set(p,'LineStyle','--')

median_inh(1)=0;
median_inh(2)=median(radiiHistory_inh(100,:));
median_inh(3)=median(radiiHistory_inh(200,:));
median_inh(4)=median(radiiHistory_inh(300,:));
median_inh(5)=median(radiiHistory_inh(400,:));
median_inh(6)=median(radiiHistory_inh(500,:));
median_inh(7)=median(radiiHistory_inh(600,:));
% p=plot(median_inh);
% set(p,'Color','red')
% set(p,'LineStyle','--')

set(gca,'FontSize',18,'LineWidth',2)
xlabel('Time (sec)')
ylabel('Radii')
xlim([1 7])
ylim([0 6.1])
set(gca,'XTickLabel',{'0','10K','20K','30K','40K','50K','60K'})
%set(gca,'yscale','log')
ch=get(gca,'Children');
set(ch(1),'LineWidth',2)
set(ch(2),'LineWidth',2)
% set(ch(3),'LineWidth',2)
% set(ch(4),'LineWidth',2)
hold off
