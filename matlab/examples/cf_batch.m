function [fraction1, fraction2, fraction3] = cf_batch(  )
%
% get xml output files (tR_[0.1-1.9]--fE_[0.90-0.98].xml
% calculate fraction of neurons w/stable connectivity radii
% between 20k and 30k seconds
% f1 - fraction of total set
% f2 - fraction of non-edge
% f3 - fraction of non-edge + neighbors

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
        [f1, f2, f3] = calc_fraction([stateoutfile, '.xml'], 100, 30000, 10, 10);
        fraction1(j, i) = f1;
        fraction2(j, i) = f2;
        fraction3(j, i) = f3;
        j = j + 1;
    end
    i = i + 1;
end

figure(1)
%surfc(x, y, fraction1)
imagesc(x, y, fraction1), axis xy
colorbar
%colormap hsv
shading interp
xlabel('Fraction of excitatory neurons')
ylabel('Target rate')
zlabel('Fraction of neurons w/stable connectivity')
title('Fraction of neurons w/stable connectivity radii (Total Set)')

figure(2)
%surfc(x, y, fraction2)
imagesc(x, y, fraction2), axis xy
colorbar
%colormap hsv
shading interp
xlabel('Fraction of excitatory neurons')
ylabel('Target rate')
zlabel('Fraction of neurons w/stable connectivity')
title('Fraction of neurons w/stable connectivity radii (Non-edge)')

figure(3)
%surfc(x, y, fraction3)
imagesc(x, y, fraction3), axis xy
colorbar
%colormap hsv
shading interp
xlabel('Fraction of excitatory neurons')
ylabel('Target rate')
zlabel('Fraction of neurons w/stable connectivity')
title('Fraction of neurons w/stable connectivity radii (Non-edge and Neighbors)')
