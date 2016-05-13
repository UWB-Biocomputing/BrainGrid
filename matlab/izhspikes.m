function izhspikes(stateoutfile)
% A matlab function to show the results of static_izh_1000.xml description file.
%    fig1 - spike raster graph
%    fig2 - populational spike activity
%    fig3 - autocorrelation of the populational spike activity

close all;

spikesProbedNeurons = double((hdf5read([stateoutfile '.h5'], 'spikesProbedNeurons'))');
spikesHistory = double((hdf5read([stateoutfile '.h5'], 'spikesHistory'))');

idx = 1;
for i=1:length(spikesProbedNeurons(1, :))
    for j=1:length(spikesProbedNeurons(:, 1))
        if (spikesProbedNeurons(j, i) ~= 0)
            firings(1, idx) = i;
            firings(2, idx) = spikesProbedNeurons(j, i);
            idx = idx + 1;
        end
    end
end

fig1 = figure(1); clf;
plot(firings(2, :), firings(1, :),'.');
xlim([0 10000])
ylim([0 1000])
xlabel('time (0.1 msec)')
ylabel('neuron number')

fig2 = figure(2); clf;
plot(spikesHistory);
xlabel('time (10 msec)')
ylabel('number of spikes')

% need Signal Processing Toolbox to call the xcorr function
fig3 = figure(3); clf;
[r, lags] = xcorr(spikesHistory);
plot(lags, r);
xlabel('lag (10 msec)')
%xlim([-40 40])

