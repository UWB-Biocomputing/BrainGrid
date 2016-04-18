function cg = plotrates(x, y, ratesHistory, neuronTypes, starterNeurons)
% PLOTRATES Plot circles showing firing rate
%   CG = PLOTRATES(X, Y, RATES, IDX) plots circles at the unit
%   locations (x, y) of radii propotional to firing rate, colored according
%   to the neuron types for each neuron idx. It retunrs the handle graphics
%   group corresponding to the circles.

% Neuron types
INH = 1;
EXC = 2;


[m, n] = size(ratesHistory);
rates = ratesHistory(m, :);
% Get the maximum rate, so we can scale the circle so they don't overlap
maxRate = max(rates);
if maxRate == 0,
    maxRate = 1;
end;

% Create and plot the circles
theta = [0:0.01:2*pi];
cg = hggroup;
for i = 1:length(x),
    rx = 0.5*rates(i)/maxRate * cos(theta);
    ry = 0.5*rates(i)/maxRate * sin(theta);
    if neuronTypes(i) == EXC,
        if find(starterNeurons == i),
            color = 'b';
        else
            color = 'g';
        end;
    else
        color = 'r';
    end;
    ch = patch(x(i)+rx, y(i)+ry, color, 'EdgeColor', 'none');
    set(ch, 'Parent', cg);
end;

% Make them circles
axis equal;
