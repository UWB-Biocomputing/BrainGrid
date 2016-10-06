function f15 = burstiness(hist)
% BURSTINESS    Plot burstiness index vs. time for spike records
%   f15 = burstiness(hist) plots burstiness and returns f15 values
%
%   hist - Burstiness history computed during simulator run

global now;

numbins = length(hist);
seglength = 300;

f15 = zeros(1, numbins-seglength+1);
fprintf('\nComputing f15 at offset: ');
for n = 1:numbins-seglength+1,
    if mod(n, 1000)==0,
        fprintf('%d', n);
    end;
    if mod(n, 10000)==0,
        fprintf('\n');
    end;
    bins = sort(hist(n:n+seglength-1)); % bin values; smallest to largest
    spikes = sum(bins); % total number of spikes in window
    sp15 = sum(bins(round(0.85*seglength):seglength));  % spikes in largest 15%
    if (sp15 == 0) && (spikes == 0),
        f15(n) = 0;
    else 
        f15(n) = sp15/spikes;
    end;
end;

plot((f15'-0.15)/0.85);
set(gca,'FontSize',18,'LineWidth',2)
xlabel('Time (s)');
ylabel('Burstiness Index');
xlim([0 now])
fprintf('\n');
