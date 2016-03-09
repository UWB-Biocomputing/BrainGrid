function [brate1, brate2] = cb_batch(  )
%
% get xml output files (tR_[0.1-1.9]--fE_[0.90-0.98].xml
% calculate average burstiness index
% brate1 - average burstiness index of 20k - 25k sec
% brate2 - average burstiness index of 25k - 30k sec


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
        [f1, f2] = calc_aburst([stateoutfile, '.xml']);
        brate1(j, i) = f1;
        brate2(j, i) = f2;
        j = j + 1;
    end
    i = i + 1;
end

figure(1)
%surfc(x, y, brate1)
imagesc(x, y, brate1), axis xy
colorbar
%colormap hsv
shading interp
xlabel('Fraction of excitatory neurons')
ylabel('Target rate')
%zlabel('Average burstiness index of 20k - 25k sec')
title('Average burstiness index of 20k - 25k sec')

figure(2)
%surfc(x, y, brate2)
imagesc(x, y, brate2), axis xy
colorbar
%colormap hsv
shading interp
xlabel('Fraction of excitatory neurons')
ylabel('Target rate')
%zlabel('Average burstiness index of 25k - 30k sec')
title('Average burstiness index of 25k - 30 sec')


