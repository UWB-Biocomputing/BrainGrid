function [fig5a, fig5b, fig5c, fig5d, fig5e, fig5f, fig5g] = growth5()
% plot distribusions of background firing rate

global spikesHistory;
global numNeurons;
global now;
global numSims;

spikesHistory10= spikesHistory/numNeurons;      % spikes/neuron in 10 ms

unit_in_second=0.01;                            % minimum unit (10ms) in second

fig5a=figure(50); clf;
if numSims >= 51
    [n1,xout1]=hist(spikesHistory10(1,500000:510000)/unit_in_second,0:0.01:0.5);
    fprintf('500000:510000 sum, mean, std\n');
    sum(n1/10000)
    % probability distribution plot
    bar(xout1,n1/10000,1)
    mean(spikesHistory10(1,500000:510000)/unit_in_second)
    std(spikesHistory10(1,500000:510000)/unit_in_second)
    [muhat,sigmahat]=normfit(spikesHistory10(1,500000:510000)/unit_in_second);
    set(gca,'FontSize',18,'LineWidth',2)
    xlim([0 0.5])
    ylim([0 0.1])
    xlabel('Spikes count(Hz) per neuron')
    ylabel('Spikes count probabilty')
end

fig5b=figure(51); clf;
if numSims >= 51
    normplot(spikesHistory10(1,500000:510000)/unit_in_second)
    set(gca,'FontSize',18,'LineWidth',2)
    ch=get(gca,'Children');
    set(ch(1),'LineWidth',2)
    set(ch(2),'LineWidth',2)
    set(ch(3),'LineWidth',2)
end

fig5c=figure(52); clf;
if numSims >= 531
    spikesHistory11=spikesHistory10(1,5300000:5310000)/unit_in_second;
    idx1=find(spikesHistory11 <= 0.5);
    idx12=find(spikesHistory11 > 0.5);
    [n2,xout2]=hist(spikesHistory11(idx1),0:0.01:0.5);
    fprintf('5300000:5310000 sum, mean, std\n');
    sum(n2/10000)
    % probability distribution plot
    bar(xout2,n2/10000,1)
    mean(spikesHistory11(idx1))
    std(spikesHistory11(idx1))
    [muhat,sigmahat]=normfit(spikesHistory11(idx1));
    set(gca,'FontSize',18,'LineWidth',2)
    xlim([0 0.5])
    ylim([0 0.1])
    xlabel('Spikes count(Hz) per neuron')
    ylabel('Spikes count probabilty')
end

% figure(29); clf;
% hist(spikesHistory11(idx12),0.5:90)

fig5d=figure(53); clf;
if numSims >= 531
    normplot(spikesHistory11(idx1))
    set(gca,'FontSize',18,'LineWidth',2)
    ch=get(gca,'Children');
    set(ch(1),'LineWidth',2)
    set(ch(2),'LineWidth',2)
    set(ch(3),'LineWidth',2)
end

% figure(28); clf;
% hist(spikesHistory10(1,1000000:1010000)/unit_in_second,0:50)
% xlim([0 50])
% ylim([0 100])

% figure(29); clf;
% hist(spikesHistory10(1,1000000:1010000)/unit_in_second)

fig5e=figure(54); clf;
if numSims >= 600
    spikesHistory12=spikesHistory10(1,5990000:6000000)/unit_in_second;
    idx2=find(spikesHistory12 <= 0.5);
    [n3,xout3]=hist(spikesHistory12(idx2),0:0.01:0.5);
    fprintf('5990000:6000000 sum, mean, std\n');
    sum(n3/10000)
    % probability distribution plot
    bar(xout3,n3/10000,1)
    mean(spikesHistory12(idx2))
    std(spikesHistory12(idx2))
    [muhat,sigmahat]=normfit(spikesHistory12(idx2));
    set(gca,'FontSize',18,'LineWidth',2)
    xlim([0 0.5])
    ylim([0 0.1])
    xlabel('Spikes count(Hz) per neuron')
    ylabel('Spikes count probabilty')
end

fig5f=figure(55); clf;
if numSims >= 600
    normplot(spikesHistory12(idx2))
    set(gca,'FontSize',18,'LineWidth',2)
    ch=get(gca,'Children');
    set(ch(1),'LineWidth',2)
    set(ch(2),'LineWidth',2)
    set(ch(3),'LineWidth',2)
end

fig5g=figure(56); clf;
spikesHistory13=spikesHistory10(1,1:now*100)/unit_in_second;
% hold on
hist(spikesHistory13,0:0.01:1000);
set(gca,'xscale','log')
set(gca,'yscale','log')
set(gca,'FontSize',18,'LineWidth',2)
xlim([0 200])
xlabel('Spikes count(Hz) per neuron')
ylabel('Number of samples')
set(gca,'XTick',[0.01 0.1 1 10 100])
% axis tight
% hold off

