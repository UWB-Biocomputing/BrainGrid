function [frate1, frate2, frate3, frate4] = ca_batch(  )
%
% get xml output files (tR_[0.1-1.9]--fE_[0.90-0.98].xml
% calculate average firing rate
% frate1 - average firing rate of 20k - 25k sec
% frate2 - average firing rate of 25k - 30k sec
% frate3 - frate1 / tR (target rate)
% frate4 - frate2 / tR (target rate)


clear;
close all;

x = [];

i = 1;
for fE = 0.90:0.01:0.98
    j = 1;
    x = [x, fE];
    y = [];
    for tR = 0.1:0.2:1.9
        y = [y, tR];
        stateoutfile = ['tR_', num2str(tR, '%1.1f'),'--fE_', num2str(fE, '%1.2f')];
        [f1, f2, f3, f4] = calc_afrate([stateoutfile, '.xml'], 100, tR);
        frate1(j, i) = f1;
        frate2(j, i) = f2;
        frate3(j, i) = f3;
        frate4(j, i) = f4;
        j = j + 1;
    end
    i = i + 1;
end

figure(1)
%surfc(x, y, frate1)
imagesc(x, y, frate1), axis xy
colorbar
%colormap hsv
shading interp
xlabel('Fraction of excitatory neurons')
ylabel('Target rate')
%zlabel('Average firing rate of 20k - 25k sec')
title('Average Firing Rate of 20k - 25k sec')

figure(2)
%surfc(x, y, frate2)
imagesc(x, y, frate2), axis xy
colorbar
%colormap hsv
shading interp
xlabel('Fraction of excitatory neurons')
ylabel('Target rate')
%zlabel('Average firing rate of 25k - 30k sec')
title('Average Firing Rate of 25k - 30k sec')

figure(3)
%surfc(x, y, frate3)
imagesc(x, y, frate3), axis xy
colorbar
%colormap hsv
shading interp
xlabel('Fraction of excitatory neurons')
ylabel('Target rate')
%zlabel('Average firing rate of 20k - 25k sec / tR (target rate)')
title('Average Firing Rate of 20k - 25k sec / tR (target rate)')

figure(4)
%surfc(x, y, frate4)
imagesc(x, y, frate4), axis xy
colorbar
%colormap hsv
shading interp
xlabel('Fraction of excitatory neurons')
ylabel('Target rate')
%zlabel('Average firing rate of 25k - 30k sec / tR (target rate)')
title('Average Firing Rate of 25k - 30 sec / tR (target rate)')
