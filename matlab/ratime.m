function [N,X] = ratime(R, bin, simlength)
% RATIME   Plot rate vs. time for spike records
%

numchannels = length(R.channel);
numbins = floor(simlength/bin);
%N = zeros(numbins, numchannels);
%X = zeros(numbins, numchannels);

for n = 1:numchannels,
    [N(n,:),X(n,:)] = hist(R.channel(n).data, numbins);
end;

N = N/bin;

plot(X', N');
xlabel('Time (s)');
ylabel('Firing rate (Hz)');

