function [f1, f2] = calc_aburst(stateoutfile)

% calculate average burstiness index
% f1 - average burstiness index of 20k - 25k sec
% f2 - average burstiness index of 25k - 30k sec

burstinessHist = readMatrix(stateoutfile, 'burstinessHist');

numbins = length(burstinessHist);
seglength = 300;

f15 = zeros(1, numbins-seglength+1);
%fprintf('\nComputing f15 at offset: ');
for n = 1:numbins-seglength+1,
%    if mod(n, 1000)==0,
%        fprintf('%d', n);
%    end;
%    if mod(n, 10000)==0,
%        fprintf('\n');
%    end;
    bins = sort(burstinessHist(n:n+seglength-1)); % bin values; smallest to largest
    spikes = sum(bins); % total number of spikes in window
    sp15 = sum(bins(round(0.85*seglength):seglength));  % spikes in largest 15%
    if (sp15 == 0) && (spikes == 0),
        f15(n) = 0;
    else
        f15(n) = sp15/spikes;
    end;
end;

% average burstiness index = sum of burstiness index in r * c matrix divide by
% number of elements in the matrix (r * c)
sIndx = 20000;
eIndx = 25000;
[r c] = size(f15(:, sIndx:eIndx));
f1 = sum(f15(1, sIndx:eIndx,:)) / c;
f1 = (f1 - 0.15) / 0.85;

sIndx = 25001;
eIndx = numbins-seglength+1;
[r c] = size(f15(:, sIndx:eIndx));
f2 = sum(f15(1, sIndx:eIndx)) / c;
f2 = (f2 - 0.15) / 0.85;

