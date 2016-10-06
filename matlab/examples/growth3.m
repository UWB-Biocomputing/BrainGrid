function [fig3a, fig3b, fig3c] = growth3()
% plot threshold voltage vs. final radii
% plot threshold voltage vs. spontaneous firing rate
global ratesHistory;
global radiiHistory;
global neuronThresh;
global starterNeurons;

finalRadii = radiiHistory(size(radiiHistory, 1), :);
starterRadii = finalRadii(starterNeurons);
% unit -> mv
starterThresh = neuronThresh(starterNeurons) * 1000;

fprintf('Plotting neuron threshold voltage vs. final radii.\n');
fig3a = figure(31); clf;
scatter(starterThresh, starterRadii, '.b');
axis tight
xlabel('Threshold voltage');
ylabel('Final radii');
title('Starter neuron threshold voltage vs. final radii');

% get average of initial 10 values
ratesAve = sum(ratesHistory(2:11, :)) / 10;
size(ratesAve)
starterRatesAve = ratesAve(starterNeurons);

fprintf('Plotting neuron threshold voltage vs. spontaneous firing rate.\n');
fig3b = figure(32); clf;
scatter(starterThresh, starterRatesAve, '.b');
axis tight
xlabel('Threshold voltage (mV)');
ylabel('Spontaneous firing rate (Hz)');
title('Starter neuron threshold voltage vs. spontaneous firing rate');

fprintf('spontaneous firing rate vs. final radii.\n');
fig3c = figure(33); clf;
scatter(starterRatesAve, starterRadii, '.b');
axis tight
xlabel('Spontaneous firing rate (Hz)');
ylabel('Final radii');
title('Starter neuron threshold voltage vs. spontaneous firing rate');

fprintf('Minimum rate');
min(starterRatesAve)
max(starterRatesAve)
mean(starterRatesAve)

%pause