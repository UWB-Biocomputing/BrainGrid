function [N,S] = sumratime(R, bin, simlength)
% SUMRATIME   Plot rate vs. time for spike records
%             (sum of individual cell)
%

edges = [0:bin:simlength];

numchannels = length(R.channel);
numbins = floor(simlength/bin) + 1;

for n = 1:numchannels,
    [N(n,:)] = histc(R.channel(n).data, edges);
end;

N = N/bin;

for m = 1:numbins,
    S(m) = sum(N(:,m));
end;

plot(edges, S);
xlabel('Time (s)');
ylabel('Sum of Firing rate (Hz)');
