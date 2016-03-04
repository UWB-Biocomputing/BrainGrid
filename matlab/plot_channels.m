function plot_channels(rates, Tsim, now)

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
