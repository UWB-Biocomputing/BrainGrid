function [f1, f2, f3, f4] = calc_afrate(stateoutfile, Tsim, tR)

% calculate average firing rate
% f1 - average firing rate of 20k - 25k sec
% f2 - average firing rate of 25k - 30k sec
% f3 - f1 / tR (target rate)
% f4 - f2 / tR (target rate)

ratesHistory = readMatrix(stateoutfile, 'ratesHistory');

% average firing rate = sum of firing rates in r * c matrix divide by
% number of elements in the matrix (r * c)
sIndx = 20000/Tsim;
eIndx = 25000/Tsim;
[r c] = size(ratesHistory(sIndx:eIndx,:));
f1 = sum(sum(ratesHistory(sIndx:eIndx,:))) / (r * c);

sIndx = 25100/Tsim;
eIndx = 30100/Tsim;
[r c] = size(ratesHistory(sIndx:eIndx,:));
f2 = sum(sum(ratesHistory(sIndx:eIndx,:))) / (r * c);

f3 = f1 / tR;
f4 = f2 / tR;
