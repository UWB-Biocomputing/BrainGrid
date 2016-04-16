function [mradii1, var1, mradii2, var2] = calc_mradii(stateoutfile, Tsim, now)

% calculate mean and variance of final radii of INH and starter neurons
% mradii1 - mean of final radii of INH neurons
% var1    - variance of final radii of INH neurons
% mradii2 - mean of final radii of starter neurons
% var2    - variance of final radii of starter neurons

INH = 1;
EXC = 2;

radiiHistory = readMatrix(stateoutfile, 'radiiHistory');
neuronTypes = readMatrix(stateoutfile, 'neuronTypes');
starterNeurons = readMatrix(stateoutfile, 'starterNeurons');
starterNeurons = starterNeurons + 1;    % 1 based indexing

mradii1 = mean(radiiHistory((now / Tsim) + 1,find(neuronTypes == INH)));
var1 = var(radiiHistory((now / Tsim) + 1,find(neuronTypes == INH)));

mradii2 = mean(radiiHistory((now / Tsim) + 1,starterNeurons));
var2 = var(radiiHistory((now / Tsim) + 1,starterNeurons));