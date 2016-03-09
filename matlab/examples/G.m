function G()

f=0:0.01:1;
g=1-2./(1+exp((0.6-f)./0.1));

fig = figure(1); clf;
plot(f,g)
set(gca,'FontSize',18,'LineWidth',2)
xlabel('Firing Rate, Fi');
ylabel('Outgrowth, G()');
ch=get(gca,'Children');
set(ch(1),'LineWidth',2)

saveas(fig, ['G1.jpg']);
