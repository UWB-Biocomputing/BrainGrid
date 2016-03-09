% radii movie

for n=1:4001,
clf;
plotradii(xloc, yloc, radiiHistory(:,n), idxNeurons);
set(gca, 'XLim', [-5 10], 'YLim', [-5 10]);
title(['t = ' num2str(n*10) ' seconds']);
frame(n) = getframe;
end;
