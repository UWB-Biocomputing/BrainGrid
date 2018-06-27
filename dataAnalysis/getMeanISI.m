%GETMEANISI return mean interspike interval for a population of p neurons
%Given h5file directory and p neurons (number of neurons in a population), 
%total number of spikes and total number of neurons are collected from 
%h5file, then mean ISI (unit: timesteps) is calculated for this population.
% 
%   Syntax: getMeanISI(h5dir, p)
%   
%   Input:  h5dir       - BrainGrid simulation result (.h5)
%           p           - number of neurons in the population of interest
%
%   Return: size        - dataset size (column size for matrix)

% Author:   Jewel Y. Lee (jewel.yh.lee@gmail.com)
% Last updated: 6/26/2018
function meanISI = getMeanISI(h5dir, p)
%h5dir = '/Users/jewellee/Desktop/thesis-work/BrainGrid/tR_1.0--fE_0.90';
spikeTimeFile = [h5dir, '/allSpikeTimeCount.csv'];
if spikeTimeFile(spikeTimeFile, 'file') == 2
    spikeTimeCount = csvread(spikeTimeFile,0,1);
else
    error(['ERROR: File', spikeTimeFile, ' is missing']);
end
totalCount = sum(spikeTimeCount);           % total number of spikes
n_bins = getH5datasetSize(h5dir, 'spikesHistory');
binSize = 100;                              % 10ms = 100 time steps
totalTimeSteps = n_bins*binSize;            % total number of time steps
n_neurons = getH5datasetSize(h5dir, 'neuronTypes'); % total number of neurons
meanISI = (totalTimeSteps/totalCount)*(n_neurons/p); 
end