function plot_channels(rates, Tsim, now)
% PLOT_CHANNELS   Raster plot
%   plot_channels(rates, Tsim, now) create raster plot
%
%   Creates a raster plot of neural activity. A point is plotted for
%   each neuron that has a non-zero firing rate. Y axis is neuron
%   number; X axis is time.
%
%   rates - firing rate history of neurons
%   Tsim - simulation length (isn't currently used)
%   now - current simulation time (so X axis will end here, rather
%         than at last time there was a non-zero firing rate)


cla reset; hold on;
[m, n] = size(rates);
for c=1:n
    st = find(rates(:,c)');
    plot(st, c*ones(1, length(st)), 'k.');
end
axis tight
if ~isempty(now)
    set(gca, 'Xlim', [0 m]);
end
