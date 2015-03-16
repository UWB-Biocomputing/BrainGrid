function [ng, rg] = plotradii(x, y, radiiHistory, neuronTypes, starterNeurons)
% PLOTRADII Plot circles showing connectivity radii
%   [NG, RG] = PLOTRADII(X, Y, RADII, IDX) plots points at the unit
%   locations (x, y) and then plots circles of radii colored according
%   to the neuron types for each neuron idx. It returns the handle graphics
%   groups corresponding to the neuron points and radius circles.

% Neuron types
INH = 1;
EXC = 2;

% Plot the unit locations and group them
nh = plot(x, y, 'k.');
ng = hggroup;
set(nh, 'Parent', ng);
hold on
axis equal

% Create and plot the circles
[m, n] = size(radiiHistory);
radii = radiiHistory(m, :);
theta = [0:0.01:2*pi];
rg = hggroup;
for i = 1:length(x),
    rx = radii(i) * cos(theta);
    ry = radii(i) * sin(theta);
    if neuronTypes(i) == EXC,
        if find(starterNeurons == i),
            style = 'b-';
        else
            style = 'g-';
        end;
    else
        style = 'r-';
    end;
    rh = plot(x(i)+rx, y(i)+ry, style);
    set(rh, 'Parent', rg);
end;
