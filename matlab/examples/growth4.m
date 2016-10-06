function [fig4a, fig4b, fig4c, fig4d, fig4f, fig4g, fig4h, fig4i] = growth4()

global spikesHistory;
global numNeurons;
global reg_peakpos;
global numSims;

spikesHistory10= spikesHistory/numNeurons;      % spikes/neuron in 10 ms

unit_in_second=0.01;                            % minimum unit (10ms) in second
len_epoch_second=100;                           % length of an epoch in second
len_epoch=len_epoch_second/unit_in_second;      % length of an epoch in minimum unit (10 ms)
num_epoch=length(spikesHistory10)/len_epoch;    % # of epoch in the simultion
len_sim_second=num_epoch*len_epoch_second;      % length of simulation in second
bs_length=70;                                   % maximum burst width
b_threash=0.005;                                % burst threashold 
num_epoch_plot=10;                              % # of epoch per unit of plot (segment)
b_shapes=zeros(num_epoch/num_epoch_plot,bs_length); % an array to store shape
%clear reg_peakpos;          % store peak positions
index_p=1;                  % index of reg_peakpos

for i=1:num_epoch/num_epoch_plot  % for each segment
%     k=1;
     s_index=(i-1)*(len_epoch*num_epoch_plot)+1;
     e_index=s_index+(len_epoch*num_epoch_plot)-1;
%     % convert ms bin into 10ms bin
%     for j=1:10:(lepoch-9)
%         % spikesHistory10 stores spikes/neuron in 10ms bin
%         spikesHistory10(k)=sum(spikesHistory01(1,[s_index+j:s_index+j+9]));
%         k=k+1;
%     end
    % find peaks (peak #, position, height, width)
%     L=findpeaks([1:size(spikesHistory10,2)],spikesHistory10,0,b_threash,1,3,3);
%     if L(1,1)~=0
%         b_count1(i)=size(L,1);      % total bursts count of the epoch
%         b_height(i)=mean(L(:,3));   % mean bursts height of the epoch
%         diff_pos=diff(L(:,2));
%         b_count2(i)=length(find(diff_pos >= 40))+1; % total bursts count of the epoch (cut out same peaks within burst)
%     else
%         b_count1(i)=0;
%         b_count2(i)=0;
%         b_height(i)=0;
%     end
    s_count(i)=sum(spikesHistory10(1,s_index:e_index)); % total spikes count of the epoch
    
    b_shape=zeros(1,bs_length); % mean burst shape of the epoch
    b_index=1;      % index in burst
    b_count3(i)=0;  % total bursts count of the epoch
    
    b_height2(i)=0; % mean burst height of the epoch
    b_width(i)=0;   % mean burst width of the epoch
    b_peakpos(i)=0; % mean peak position in bursts of the epoch
    b_scount(i)=0;  % mean spikes count/burst of the epoch
    
    reg_b_index=1;
    clear reg_b_height2; % data burst height of the epoch
    clear reg_b_width;   % data burst width of the epoch
    clear reg_b_peakpos; % data peak position in bursts of the epoch
    clear reg_b_scount;  % data spikes count/burst of the epoch  
    
    b_height2_std(i)=0; % std burst height of the epoch
    b_width_std(i)=0;   % std burst width of the epoch
    b_peakpos_std(i)=0; % std peak position in bursts of the epoch
    b_scount_std(i)=0;  % std spikes count/burst of the epoch
    j=0;            % pointer in segment
    
    % analyse data in one segment
    while j<len_epoch*num_epoch_plot % while one segment
        b_height_tmp=0;
        b_peakpos_tmp=0;
        b_scount_tmp=0;
        % while in one burst
        while j<len_epoch*num_epoch_plot & spikesHistory10(s_index+j)>=b_threash & b_index<=bs_length
            if b_index==1 & j~=0
                b_shape(b_index)=b_shape(b_index)+spikesHistory10(s_index+j-1);
                b_index=b_index+1;
                b_startpos=j;
            end
            b_shape(b_index)=b_shape(b_index)+spikesHistory10(s_index+j);
            if spikesHistory10(s_index+j) > b_height_tmp
                b_height_tmp=spikesHistory10(s_index+j);
                b_peakpos_tmp=b_index;
            end
            b_scount(i)=b_scount(i)+spikesHistory10(s_index+j);
            b_scount_tmp=b_scount_tmp+spikesHistory10(s_index+j);
            b_index=b_index+1;
            j=j+1;
        end % end of while (burst)
        if b_index>1
            reg_b_index=reg_b_index+1;
            reg_b_height2(reg_b_index)=b_height_tmp;
            reg_b_width(reg_b_index)=b_index;
            reg_b_peakpos(reg_b_index)=b_peakpos_tmp;
            reg_b_scount(reg_b_index)=b_scount_tmp;
            
            b_count3(i)=b_count3(i)+1;
            b_height2(i)=b_height2(i)+b_height_tmp;
            b_width(i)=b_width(i)+b_index;
            b_peakpos(i)=b_peakpos(i)+b_peakpos_tmp;
            b_index=1;
            reg_peakpos(index_p)=s_index+b_startpos+b_peakpos_tmp;
            index_p=index_p+1;
        end
        j=j+1;
    end % end of while (segment)
    
    if b_count3(i)>0
        b_height2_std(i)=std(reg_b_height2);
        b_width_std(i)=std(reg_b_width);
        b_peakpos_std(i)=std(reg_b_peakpos);
        b_scount_std(i)=std(reg_b_scount);
        
        b_shape=b_shape/b_count3(i);
        b_height2(i)=b_height2(i)/b_count3(i);
        b_scount(i)=b_scount(i)/b_count3(i);
        b_width(i)=b_width(i)/b_count3(i);
        b_peakpos(i)=b_peakpos(i)/b_count3(i);   
        b_peakpos2(i)=find(b_shape==max(b_shape),1);
    end
    reg_fanofactor(i)=var(spikesHistory(1,s_index:e_index))/mean(spikesHistory(1,s_index:e_index));
    %reg_fanofactor(i)=var(spikesHistory(1,s_index:e_index));
    b_shapes(i,:)=b_shape;

%     if i==59 
% %         figure(40); clf;
% %         hist(diff_pos,100);
%         
%         fig4h = figure(31); clf;         
%         %plot([s_index:e_index],spikesHistory10(1,s_index:e_index));
%         % An exmaple of a burst for 'tR_0_1--fE_0_90_historyDump'
%         %plot([0:1000],spikesHistory10(1,0970000:0971000)/unit_in_second);
%         plot(b_shape/unit_in_second)
%         set(gca,'XTickLabel',{'0','0.1','0.2','0.3','0.4'})
%         set(gca,'FontSize',18,'LineWidth',2)    
%         xlabel('Time (sec)')
%         ylabel('Hz per neuron')
%         ylim([0 160])
%         ch=get(gca,'Children');
%         set(ch(1),'LineWidth',2)
%         saveas(fig4h, 'ex-bshape1.jpg');
%     end    
%     
%     if i==53   
% %         figure(42); clf;
% %         hist(diff_pos,100);
%         
%         fig4i = figure(33); clf;
%         %plot([s_index:e_index],spikesHistory10(1,s_index:e_index));
%         % An exmaple of a burst for 'tR_0_1--fE_0_90_historyDump'
%         plot([0:40],spikesHistory10(1,5242260:5242300)/unit_in_second);
%         set(gca,'XTickLabel',{'0','0.1','0.2','0.3','0.4'})
%         set(gca,'FontSize',18,'LineWidth',2)    
%         xlabel('Time (sec)')
%         ylabel('Hz per neuron')
%         ylim([0 160])
%         ch=get(gca,'Children');
%         set(ch(1),'LineWidth',2)
%         saveas(fig4i, 'ex-bshape2.jpg');
%     end

%     if i==14   
%         figure(44); clf;
%         hist(diff_pos,100);
%     end
    
    %clear spikesHistory10;
end % for each segment

% x0=75;
% x1=130;
% x2=147;

% Bursts count per sec
fig4a = figure(40); clf;
hold on
% plot(b_count1,'g')
% plot(b_count2,'b')
% plot(b_count3,'r')
plot(b_count3/(len_epoch_second*num_epoch_plot),'b')
set(gca,'XTickLabel',{'0','10000','20000','30000','40000','50000','60000'})
%scatter([x0 x1 x2],[b_count3(x0) b_count1(x1) b_count1(x2)],'.r')
set(gca,'FontSize',18,'LineWidth',2)
xlabel('Time (sec)')
ylabel('Bursts count per sec')
xlim([0 60])
ylim([0 0.65])
ch=get(gca,'Children');
set(ch(1),'LineWidth',2)
%title('Bursts count plot')
hold off

% Mean burst height (Hz per neuron)
fig4b = figure(41); clf;
hold on
% plot(b_height,'g')
% plot(b_height2,'r')
plot(b_height2/unit_in_second,'b')
%errorbar(b_height2/unit_in_second,b_height2_std/unit_in_second,'b')
set(gca,'XTickLabel',{'0','10000','20000','30000','40000','50000','60000'})
%scatter([x0 x1 x2],[b_height(x0) b_height(x1) b_height(x2)],'.r')
set(gca,'FontSize',18,'LineWidth',2)
xlabel('Time (sec)')
ylabel('Mean burst height (Hz per neuron)')
xlim([0 60])
ylim([0 180])
ch=get(gca,'Children');
set(ch(1),'LineWidth',2)
%title('Burst height plot')
hold off

% fprintf('burst height: 34,000, 39,000, 60,000')
% b_height2(34)/unit_in_second
% if numSims >= 390
%     b_height2(39)/unit_in_second
% end
% if numSims >= 600
%     b_height2(60)/unit_in_second
% end

% Mean spikes count (Hz per neuron)
fig4c = figure(42); clf;
hold on
plot(s_count/(len_epoch_second*num_epoch_plot),'b')
set(gca,'XTickLabel',{'0','10000','20000','30000','40000','50000','60000'})
%scatter([x0 x1 x2],[s_count(x0) s_count(x1) s_count(x2)],'.r')
set(gca,'FontSize',18,'LineWidth',2)
xlabel('Time (sec)')
ylabel('Mean spikes count (Hz per neuron)')
xlim([0 60])
ylim_max=2.1;
if max(s_count/(len_epoch_second*num_epoch_plot)) > ylim_max
    ylim_max=120;
end
ylim([0 ylim_max])
ch=get(gca,'Children');
set(ch(1),'LineWidth',2)
%title('Spikes count plot')
hold off

% Mean burst width and peak position (sec)
fig4d = figure(43); clf;
hold on
%plot(b_width*unit_in_second,'b')
plot(b_width*unit_in_second,'k')
%plot(b_peakpos*unit_in_second,'r');
p=plot(b_peakpos*unit_in_second,'k');
set(p,'LineStyle','--')
%plot(b_peakpos2*unit_in_second,'g');
%errorbar(b_width*unit_in_second,b_width_std*unit_in_second,'b')
%errorbar(b_peakpos*unit_in_second,b_peakpos_std*unit_in_second,'r')
set(gca,'XTickLabel',{'0','10000','20000','30000','40000','50000','60000'})
%scatter([x0 x1 x2],[b_width(x0) b_width(x1) b_width(x2)],'.r')
set(gca,'FontSize',18,'LineWidth',2)
xlabel('Time (sec)')
ylabel('Mean burst width and peak position (sec)')
xlim([0 60])
ylim([0 0.6])
ch=get(gca,'Children');
set(ch(1),'LineWidth',2)
set(ch(2),'LineWidth',2)
%title('Burst shape plot')
hold off

% fprintf('burst width: 34,000, 39,000, 60,000')
% b_width(34)*unit_in_second
% if numSims >= 390
%     b_width(39)*unit_in_second
% end
% if numSims >= 600
%     b_width(60)*unit_in_second
% end

% fig4e = figure(17); clf;
% hold on
% plot(b_peakpos./b_width,'b')
% %scatter([x0 x1 x2],[b_width(x0) b_width(x1) b_width(x2)],'.r')
% hold off

% Mean spikes count per burst
fig4f = figure(44); clf;
hold on
plot(b_scount,'b')
%errorbar(b_scount,b_scount_std,'b')
set(gca,'XTickLabel',{'0','10000','20000','30000','40000','50000','60000'})
%scatter([x0 x1 x2],[b_scount(x0) b_scount(x1) b_scount(x2)],'.r')
set(gca,'FontSize',18,'LineWidth',2)
xlabel('Time (s)')
ylabel('Mean spikes count per neuron per burst')
xlim([0 60])
ylim([0 7])
ch=get(gca,'Children');
set(ch(1),'LineWidth',2)
%title('Spikes count in bursts plot')
hold off

% Mean burst height (Hz per neuron) (3D plot)
fig4g = figure(45); clf;
surfl(b_shapes/unit_in_second)
set(gca,'XTickLabel',{'0','200','400','600','800'})
set(gca,'FontSize',18,'LineWidth',2)
xlabel('Mean burst width (ms)')
ylabel('Time (Ksec)')
zlabel('Mean burst height (Hz per neuron)')
ylim([0 60])
zlim([0 150])
shading interp
grid on
view(-37.5+70,60)
%colormap('cool')
colormap('gray')
%caxis auto
%colorbar

% Fano factor
fig4h = figure(46); clf;
plot(reg_fanofactor,'b')
set(gca,'XTickLabel',{'0','10000','20000','30000','40000','50000','60000'})
set(gca,'FontSize',18,'LineWidth',2)
xlabel('Time (sec)')
ylabel('Fano factor of spikes')
ch=get(gca,'Children');
set(ch(1),'LineWidth',2)

% Spectrogram of spike counts in 10ms bins (spikesHistory10)
fig4i=figure(47); clf;
window=256*32;
noverlap=window/2;
nfft=window;
fs=100;
[S,F,T,P]=spectrogram(spikesHistory10,window,noverlap,nfft,fs);
surf(T,F,10*log10(P),'edgecolor','none'); axis tight; 
view(-37.5+70,60)
xlabel('Time (Seconds)'); ylabel('Hz');
