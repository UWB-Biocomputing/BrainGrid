function f15 = burstiness(R, simlength)
% BURSTINESS   Plot burstiness index vs. time for spike records
%

numchannels = length(R.channel);
numbins = floor(simlength);
seglength = 300;

% Compute the 1s bin counts for each channel; sum them into a aggregate
% set of counts for the entire net
tot = zeros(1, numbins);
fprintf('Processing channel: ');
for n = 1:numchannels,
    fprintf('%d ', n);
    if mod(n,15)==0,
        fprintf('\n');
    end;
    [N, X] = hist(R.channel(n).data, numbins);
    tot = tot + N;
end;

f15 = zeros(1, numbins-seglength+1);
fprintf('\nComputing f15 at offset: ');
for n = 1:numbins-seglength+1,
    if mod(n,1000)==0,
        fprintf('%d ', n);
    end;
    if mod(n,10000)==0,
        fprintf('\n');
    end;
    bins = sort(tot(n:n+seglength-1)); % bin values; smallest to largest
    spikes = sum(bins);   % total number of spikes in window
    sp15 = sum(bins(round(0.85*seglength):seglength)); % spikes in largest 15%
    f15(n) = sp15/spikes;
end;

plot(X(1:numbins-seglength+1)', (f15'-0.15)/0.85);
xlabel('Time (s)');
ylabel('Burstiness Index');
fprintf('\n');


