function [f1, f2, f3] = calc_fraction(stateoutfile, Tsim, now, xlen, ylen)

% calculate fraction of neurons w/stable connectivity radii
% between (now - 10k) and now seconds
% f1 - fraction of total set
% f2 - fraction of non-edge
% f3 - fraction of non-edge + neighbors

% if delta < minDelta then it seems to be stable
minDelta = 0.5;

radiiHistory = readMatrix(stateoutfile, 'radiiHistory');
xloc = readMatrix(stateoutfile, 'xloc');
yloc = readMatrix(stateoutfile, 'yloc');

% get delta between (now - 10k) and now seconds
delta = ( radiiHistory( now / Tsim, : ) - radiiHistory( ( now - 10000 ) / Tsim, : ) );

% count stable neurons of total set
stableNeurons = find( delta < minDelta );
c1 = length( stableNeurons );

% fraction of neurons w/stable connection of total set
f1 = c1 / length( delta );

% count stable neurons of non-edge
c2 = 0;
for i = 1:length( stableNeurons )
    if ~(xloc(stableNeurons(i)) == 0 || xloc(stableNeurons(i)) == xlen-1 ...
        || yloc(stableNeurons(i)) == 0 || yloc(stableNeurons(i)) == ylen-1)
        c2 = c2 + 1;
    end
end

% fraction of neurons w/stable connection of non-edge
f2 = c2 / ( (xlen - 2) * (ylen - 2) );

% count stable neurons of non-edge + neighbors
c3 = 0;
for i = 1:length( stableNeurons )
    if ~(xloc(stableNeurons(i)) == 0 || xloc(stableNeurons(i)) == xlen-1 ...
        || xloc(stableNeurons(i)) == 1 || xloc(stableNeurons(i)) == xlen-2 ...
        || yloc(stableNeurons(i)) == 0 || yloc(stableNeurons(i)) == ylen-1 ...
        || yloc(stableNeurons(i)) == 1 || yloc(stableNeurons(i)) == ylen-2)
        c3 = c3 + 1;
    end
end

% fraction of neurons w/stable connection of non-edge + neighbors
f3 = c3 / ( (xlen - 4) * (ylen - 4) );
