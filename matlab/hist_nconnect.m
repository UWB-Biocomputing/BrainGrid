function hist_nconnect(x, y, radiiHistory)
% HIST_NCONNECT plots histgram of number of synapses (connections)
% of each neuron
%
[m, n] = size(radiiHistory);
radii = radiiHistory(m, :);

nNeurons = length(radii)
nConnect = zeros(size(radii));

for i = 1:nNeurons
    px0 = x(i);
    py0 = y(i);
    for j = 1:nNeurons
        if i ~= j
            px1 = x(j); 
            py1 = y(j);   
            dist2 = (px1 - px0)^2 + (py1 - py0)^2;
            if dist2 < (radii(i) + radii(j))^2;
                nConnect(i) = nConnect(i) + 1;
            end
        end
    end
end

fprintf('mean number of connections.\n')
mean(nConnect)

x=0:5:80;
hist(nConnect,x);
set(gca,'FontSize',18,'LineWidth',2)
xlim([0 80])
xlabel('Number of connections')
ylabel('Neurons count')