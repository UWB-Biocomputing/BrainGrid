function [fig6a, fig6b, fig6c, fig6d, fig6e, fig6f, fig6g, fig6h, fig6i, fig6j, fig6k] = growth6()
% plot distributions of IBIs

global reg_peakpos;
global numSims;

unit_in_second=0.01;        % minimum unit (10ms) in second
idx1=intersect(find(reg_peakpos>=1000000),find(reg_peakpos<=6000000));
diff_peakpos=diff(sort(reg_peakpos(idx1)));
max_IBI=ceil(max(diff_peakpos*unit_in_second));
nblocks=30;
bin_size_hist=max_IBI/nblocks;          % bin size of histgram in 10ms    

% IBI between 10,000-20,000 sec
fig6a=figure(60); clf;
idx1=intersect(find(reg_peakpos>=1000000),find(reg_peakpos<=2000000));
diff_peakpos=diff(sort(reg_peakpos(idx1)));
% idx2=intersect(find(diff_peakpos>=0),find(diff_peakpos<=1000));
% [n,xout]=hist(diff_peakpos(idx2)*unit_in_second,30);
n_bin=((max(diff_peakpos)-min(diff_peakpos))*unit_in_second)/bin_size_hist;
[n,xout]=hist(diff_peakpos*unit_in_second,n_bin);
IBI_sum(1)=sum(n/length(diff_peakpos));
IBI_mean(1)=mean(diff_peakpos*unit_in_second)
IBI_std(1)=std(diff_peakpos*unit_in_second)
IBI_max(1)=max(diff_peakpos*unit_in_second)
IBI_min(1)=min(diff_peakpos*unit_in_second)
%[n,xout]=hist(diff_peakpos(idx)*unit_in_second);
bar(xout,n/length(diff_peakpos),1)
set(gca,'FontSize',18,'LineWidth',2)
xlabel('Inter burst interval (sec)')
ylabel('IBI probability')
xlim([0 max_IBI])
ylim([0 0.4])
fprintf('GEV distribution parameters(10,000-20,000).\n');
[paramEsts,pci]=mle(diff_peakpos*unit_in_second,'distribution','gev')
IBI_scale(1)=paramEsts(2);
IBI_loc(1)=paramEsts(3);
IBI_scale_l(1)=pci(1,2);
IBI_scale_h(1)=pci(2,2);
IBI_loc_l(1)=pci(1,3);
IBI_loc_h(1)=pci(2,3);

% IBI between 20,000-30,000 sec
fig6b=figure(61); clf;
idx1=intersect(find(reg_peakpos>=2000000),find(reg_peakpos<=3000000));
diff_peakpos=diff(sort(reg_peakpos(idx1)));
% idx2=intersect(find(diff_peakpos>=0),find(diff_peakpos<=1000));
% [n,xout]=hist(diff_peakpos(idx2)*unit_in_second,30);
n_bin=((max(diff_peakpos)-min(diff_peakpos))*unit_in_second)/bin_size_hist;
[n,xout]=hist(diff_peakpos*unit_in_second,n_bin);
IBI_sum(2)=sum(n/length(diff_peakpos));
IBI_mean(2)=mean(diff_peakpos*unit_in_second)
IBI_std(2)=std(diff_peakpos*unit_in_second)
IBI_max(2)=max(diff_peakpos*unit_in_second)
IBI_min(2)=min(diff_peakpos*unit_in_second)
%[n,xout]=hist(diff_peakpos(idx)*unit_in_second);
bar(xout,n/length(diff_peakpos),1)
set(gca,'FontSize',18,'LineWidth',2)
xlabel('Inter burst interval (sec)')
ylabel('IBI probability')
xlim([0 max_IBI])
ylim([0 0.4])
fprintf('GEV distribution parameters(20,000-30,000).\n');
[paramEsts,pci]=mle(diff_peakpos*unit_in_second,'distribution','gev')
IBI_scale(2)=paramEsts(2);
IBI_loc(2)=paramEsts(3);
IBI_scale_l(2)=pci(1,2);
IBI_scale_h(2)=pci(2,2);
IBI_loc_l(2)=pci(1,3);
IBI_loc_h(2)=pci(2,3);

% IBI between 30,000-40,000 sec
fig6c=figure(62); clf;
if numSims >= 400
    idx1=intersect(find(reg_peakpos>=3000000),find(reg_peakpos<=4000000));
    diff_peakpos=diff(sort(reg_peakpos(idx1)));
    % idx2=intersect(find(diff_peakpos>=0),find(diff_peakpos<=1000));
    % [n,xout]=hist(diff_peakpos(idx2)*unit_in_second,30);
    n_bin=((max(diff_peakpos)-min(diff_peakpos))*unit_in_second)/bin_size_hist;
    [n,xout]=hist(diff_peakpos*unit_in_second,n_bin);
    IBI_sum(3)=sum(n/length(diff_peakpos));
    IBI_mean(3)=mean(diff_peakpos*unit_in_second)
    IBI_std(3)=std(diff_peakpos*unit_in_second)
    IBI_max(3)=max(diff_peakpos*unit_in_second)
    IBI_min(3)=min(diff_peakpos*unit_in_second)
    %[n,xout]=hist(diff_peakpos(idx)*unit_in_second);
    bar(xout,n/length(diff_peakpos),1)
    set(gca,'FontSize',18,'LineWidth',2)
    xlabel('Inter burst interval (sec)')
    ylabel('IBI probability')
    xlim([0 max_IBI])
    ylim([0 0.4])
end
fprintf('GEV distribution parameters(30,000-40,000).\n');
[paramEsts,pci]=mle(diff_peakpos*unit_in_second,'distribution','gev')
IBI_scale(3)=paramEsts(2);
IBI_loc(3)=paramEsts(3);
IBI_scale_l(3)=pci(1,2);
IBI_scale_h(3)=pci(2,2);
IBI_loc_l(3)=pci(1,3);
IBI_loc_h(3)=pci(2,3);

% IBI between 40,000-50,000 sec
fig6d=figure(63); clf;
if numSims >= 500
    idx1=intersect(find(reg_peakpos>=4000000),find(reg_peakpos<=5000000));
    diff_peakpos=diff(sort(reg_peakpos(idx1)));
    % idx2=intersect(find(diff_peakpos>=0),find(diff_peakpos<=1000));
    % [n,xout]=hist(diff_peakpos(idx2)*unit_in_second,30);
    n_bin=((max(diff_peakpos)-min(diff_peakpos))*unit_in_second)/bin_size_hist;
    [n,xout]=hist(diff_peakpos*unit_in_second,n_bin);
    IBI_sum(4)=sum(n/length(diff_peakpos));
    IBI_mean(4)=mean(diff_peakpos*unit_in_second)
    IBI_std(4)=std(diff_peakpos*unit_in_second)
    IBI_max(4)=max(diff_peakpos*unit_in_second)
    IBI_min(4)=min(diff_peakpos*unit_in_second)
    %[n,xout]=hist(diff_peakpos(idx)*unit_in_second);
    bar(xout,n/length(diff_peakpos),1)
    set(gca,'FontSize',18,'LineWidth',2)
    xlabel('Inter burst interval (sec)')
    ylabel('IBI probability')
    xlim([0 max_IBI])
    ylim([0 0.4])
end
fprintf('GEV distribution parameters(40,000-50,000).\n');
[paramEsts,pci]=mle(diff_peakpos*unit_in_second,'distribution','gev')
IBI_scale(4)=paramEsts(2);
IBI_loc(4)=paramEsts(3);
IBI_scale_l(4)=pci(1,2);
IBI_scale_h(4)=pci(2,2);
IBI_loc_l(4)=pci(1,3);
IBI_loc_h(4)=pci(2,3);

% IBI between 50,000-60,000 sec
fig6e=figure(64); clf;
if numSims >= 600
    idx1=intersect(find(reg_peakpos>=5000000),find(reg_peakpos<=6000000));
    diff_peakpos=diff(sort(reg_peakpos(idx1)));
    % idx2=intersect(find(diff_peakpos>=0),find(diff_peakpos<=1000));
    % [n,xout]=hist(diff_peakpos(idx2)*unit_in_second,30);
    n_bin=((max(diff_peakpos)-min(diff_peakpos))*unit_in_second)/bin_size_hist;
    [n,xout]=hist(diff_peakpos*unit_in_second,n_bin);
    IBI_sum(5)=sum(n/length(diff_peakpos));
    IBI_mean(5)=mean(diff_peakpos*unit_in_second)
    IBI_std(5)=std(diff_peakpos*unit_in_second)
    IBI_max(5)=max(diff_peakpos*unit_in_second)
    IBI_min(5)=min(diff_peakpos*unit_in_second)
    %[n,xout]=hist(diff_peakpos(idx)*unit_in_second);
    bar(xout,n/length(diff_peakpos),1)
    set(gca,'FontSize',18,'LineWidth',2)
    xlabel('Inter burst interval (sec)')
    ylabel('IBI probability')
    xlim([0 max_IBI])
    ylim([0 0.4])
end
% 
% 
% mean and std plot
fig6f=figure(65); clf;
hold on
plot(IBI_mean)
p=plot(IBI_std)
set(gca,'FontSize',18,'LineWidth',2)
set(p,'Color','red')
xlabel('Time (sec)')
ylabel('IBI mean/STD (sec)')
xlim([1 5])
set(gca,'XTickLabel',{'10K-20K','20K-30K','30K-40K','40K-50K','50K-60K'})
ch=get(gca,'Children');
set(ch(1),'LineWidth',2)
set(ch(2),'LineWidth',2)
hold off
% 
% pdf plot overlapped IBI between 50,000-60,000 sec
fig6g=figure(66); clf
hold on
bin_size_hist=1;
n_bin=((max(diff_peakpos)-min(diff_peakpos))*unit_in_second)/bin_size_hist;
[n,xout]=hist(diff_peakpos*unit_in_second,n_bin);
bar(xout,n/length(diff_peakpos),1)
set(gca,'FontSize',18,'LineWidth',2)
xlabel('Inter burst interval (sec)')
ylabel('IBI probability')
xlim([0 max_IBI])
%ylim([0 0.5])
fprintf('GEV distribution parameters(50,000-60,000).\n');
[paramEsts,pci]=mle(diff_peakpos*unit_in_second,'distribution','gev')
IBI_scale(5)=paramEsts(2);
IBI_loc(5)=paramEsts(3);
IBI_scale_l(5)=pci(1,2);
IBI_scale_h(5)=pci(2,2);
IBI_loc_l(5)=pci(1,3);
IBI_loc_h(5)=pci(2,3);

x=linspace(0,max_IBI);
y1=gevpdf(x,paramEsts(1),paramEsts(2),paramEsts(3));
y2=gevpdf(x,pci(1,1),pci(1,2),pci(1,3));
y3=gevpdf(x,pci(2,1),pci(2,2),pci(2,3));
line(x,y1,'Color','b')
line(x,y2,'Color','r')
line(x,y3,'Color','g')
ch=get(gca,'Children');
set(ch(1),'LineWidth',2)
set(ch(2),'LineWidth',2)
set(ch(3),'LineWidth',2)
hold off
% 
% 
% scale and location plot
fig6h=figure(67); clf;
hold on
errorbar([1 2 3 4 5],IBI_loc,IBI_loc-IBI_loc_l,IBI_loc_h-IBI_loc)
p=errorbar([1 2 3 4 5],IBI_scale,IBI_scale-IBI_scale_l,IBI_scale_h-IBI_scale)
%set(p,'Color','red')
set(p,'LineStyle','--')
set(gca,'FontSize',18,'LineWidth',2)
xlabel('Time (sec)')
ylabel('IBI location/scale (sec)')
xlim([0.5 5.5])
ylim_max=5;
if ylim_max < max(IBI_loc)
    ylim_max=40;
end
ylim([0 ylim_max])
set(gca,'XTickLabel',{'10K-20K','20K-30K','30K-40K','40K-50K','50K-60K'})
%set(gca,'yscale','log')
ch=get(gca,'Children');
set(ch(1),'LineWidth',2)
set(ch(2),'LineWidth',2)
hold off

% Spectrum of inter-burst intervals between 50,000-60,000 sec
fig6i = figure(68); clf;
NFFT = 512;
L = length(diff_peakpos);
Fs = 1;
Y = fft(diff_peakpos,NFFT)/L;
f = Fs/2*linspace(0,1,NFFT/2+1);

% Plot single-sided power spectrum.
hold on
plot(f(2:end),abs(Y(2:NFFT/2+1)).^2) 
set(gca,'FontSize',18,'LineWidth',2)
xlabel('Frequency (order^{-1})')
ylabel('|Y(f)|^2')
set(gca,'yscale','log')
ch=get(gca,'Children');
set(ch(1),'LineWidth',2)
hold off

% First order return map
fig6j = figure(69); clf;
numInts = length(diff_peakpos);
plot(diff_peakpos(1:numInts-1)*unit_in_second,diff_peakpos(2:numInts)*unit_in_second,'.')
set(gca,'FontSize',18,'LineWidth',2)
ch=get(gca,'Children');
xlabel('Burst Interval i (sec)')
ylabel('Burst Interval i+1 (sec)')

% First order return map (3D)
fig6k = figure(70); clf;
numInts = length(diff_peakpos);
scatter3(diff_peakpos(1:numInts-1)*unit_in_second,diff_peakpos(2:numInts)*unit_in_second,1:numInts-1,5)
view(-80,30)
set(gca,'FontSize',18,'LineWidth',2)
ch=get(gca,'Children');
xlabel('Burst Interval i (sec)')
ylabel('Burst Interval i+1 (sec)')
