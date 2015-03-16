function batch(stateoutfile)
% batch.m
%
% get xml output files (tR_[0.1-1.9]--fE_[0.90-0.98].xml
% create fig1 - fig5 and save the figures as files
% tR_[0.1-1.9]--fE_[0.90-0.98]_fig[1-5].jpg
%
close all;

global ratesHistory;
global radiiHistory;
global burstinessHist;
global spikesHistory;
global xloc;
global yloc;
global neuronTypes;
global neuronThresh;
global starterNeurons;      % 1 based indexing
global now;                 % read now from history dump
global Tsim;                % length of an epoch
global numSims;             % number of epochs
global numNeurons;
global xlen;
global ylen;
global INH;
global EXC;
global reg_peakpos;
global fColor;
global fShowDIV;            % true if show DIV
global sSec2Div;            % convert simulation sec to virtual DIV

INH = 1;
EXC = 2;

fColor=1;
fShowDIV=0;

ratesHistory = readMatrix([stateoutfile '.xml'], 'ratesHistory');
radiiHistory = readMatrix([stateoutfile '.xml'], 'radiiHistory');
burstinessHist = readMatrix([stateoutfile '.xml'], 'burstinessHist');
spikesHistory = readMatrix([stateoutfile '.xml'], 'spikesHistory');
xloc = readMatrix([stateoutfile '.xml'], 'xloc');
yloc = readMatrix([stateoutfile '.xml'], 'yloc');
neuronTypes = readMatrix([stateoutfile '.xml'], 'neuronTypes');
neuronThresh = readMatrix([stateoutfile '.xml'], 'neuronThresh');
starterNeurons = readMatrix([stateoutfile '.xml'], 'starterNeurons') + 1;    % 1 based indexing
now = readMatrix([stateoutfile '.xml'], 'simulationEndTime');    % read now from history dump
Tsim = readMatrix([stateoutfile '.xml'], 'Tsim');                % read Tsim from history dump
numSims = now / Tsim;
numNeurons = size(ratesHistory, 2);
xlen = sqrt(numNeurons);
ylen = xlen;
reg_peakpos(1)=0;
if fShowDIV
    sSec2Div = 40/(60*60*24);
else
    sSec2Div=1;
end

%for fE = 0.90:0.01:0.98
 %   for tR = 0.1:0.2:1.9
	%tR = 1.9;
	%fE = 0.90;
%       stateoutfile = ['tR_', num2str(tR, '%0.1f'),'--fE_', num2str(fE, '%1.2f')];
        %[fig1a] = growth1();
        [fig1, fig2, fig3, fig4, fig5, fig6] = growth2();
        [fig2a, fig2b, fig2c, fig2d, fig2e, fig2f, fig2g] = growth2a();
        [fig3a, fig3b, fig3c] = growth3();
        [fig4a, fig4b, fig4c, fig4d, fig4f, fig4g, fig4h, fig4i] = growth4();
        [fig5a, fig5b, fig5c, fig5d, fig5e, fig5f, fig5g, fig5h, fig5i] = growth5();
        [fig6a, fig6b, fig6c, fig6d, fig6e, fig6f, fig6g] = growth6();
        
        % save figures
        fprintf('Saving figures\n');

        saveas(fig1, [stateoutfile, '_fig1.jpg']);
        saveas(fig2, [stateoutfile, '_fig2.jpg']);
        saveas(fig3, [stateoutfile, '_fig3.jpg']);
        saveas(fig4, [stateoutfile, '_fig4.jpg']);
        saveas(fig5, [stateoutfile, '_fig5.jpg']);
        saveas(fig6, [stateoutfile, '_fig6.jpg']);
       
        saveas(fig2a, [stateoutfile, '_fig2a.jpg']);
        saveas(fig2b, [stateoutfile, '_fig2b.jpg']);
        saveas(fig2c, [stateoutfile, '_fig2c.jpg']);
        saveas(fig2d, [stateoutfile, '_fig2d.jpg']);
        saveas(fig2e, [stateoutfile, '_fig2e.jpg']);
        saveas(fig2f, [stateoutfile, '_fig2f.jpg']);
        saveas(fig2g, [stateoutfile, '_fig2g.jpg']);
        
        saveas(fig3a, [stateoutfile, '_fig3a.jpg']);
        saveas(fig3b, [stateoutfile, '_fig3b.jpg']);
        saveas(fig3c, [stateoutfile, '_fig3c.jpg']);
        
        saveas(fig4a, [stateoutfile, '_fig4a.jpg']);
        saveas(fig4b, [stateoutfile, '_fig4b.jpg']);
        saveas(fig4c, [stateoutfile, '_fig4c.jpg']);
        saveas(fig4d, [stateoutfile, '_fig4d.jpg']);
        saveas(fig4f, [stateoutfile, '_fig4f.jpg']);
        saveas(fig4g, [stateoutfile, '_fig4g.jpg']);
        saveas(fig4h, [stateoutfile, '_fig4h.jpg']);
        saveas(fig4i, [stateoutfile, '_fig4i.jpg']);

        saveas(fig5a, [stateoutfile, '_fig5a.jpg']);
        saveas(fig5b, [stateoutfile, '_fig5b.jpg']);
        saveas(fig5c, [stateoutfile, '_fig5c.jpg']);
        saveas(fig5d, [stateoutfile, '_fig5d.jpg']);
        saveas(fig5e, [stateoutfile, '_fig5e.jpg']);
        saveas(fig5f, [stateoutfile, '_fig5f.jpg']);
        saveas(fig5g, [stateoutfile, '_fig5g.jpg']);
        saveas(fig5h, [stateoutfile, '_fig5h.jpg']);
        saveas(fig5i, [stateoutfile, '_fig5i.jpg']);
        
        saveas(fig6a, [stateoutfile, '_fig6a.jpg']);
        saveas(fig6b, [stateoutfile, '_fig6b.jpg']);
        saveas(fig6c, [stateoutfile, '_fig6c.jpg']);
        saveas(fig6d, [stateoutfile, '_fig6d.jpg']);
        saveas(fig6e, [stateoutfile, '_fig6e.jpg']);
        saveas(fig6f, [stateoutfile, '_fig6f.jpg']);
        sName={'g','h','i','j','k','l','m','n'};
        for i=1:length(fig6g)
            saveas(fig6g(i), char(strcat(stateoutfile,'_fig6',sName(i),'.jpg')));
        end
        %pause
        
        % close figures
        fprintf('Closing figures\n');
        close(fig1);
        close(fig2);
        close(fig3);
        close(fig4);
        close(fig5);
        close(fig6);
       
        close(fig2a);
        close(fig2b);
        close(fig2c);
        close(fig2d);
        close(fig2e);
        close(fig2f);
        close(fig2g);
        
        close(fig3a);
        close(fig3b);
        close(fig3c);
        
        close(fig4a);
        close(fig4b);
        close(fig4c);
        close(fig4d);
        close(fig4f);
        close(fig4g);
        close(fig4h);
        close(fig4i);
        
        close(fig5a);
        close(fig5b);
        close(fig5c);
        close(fig5d);
        close(fig5e);
        close(fig5f);         
        close(fig5g);
        close(fig5h);
        close(fig5i);

        close(fig6a);
        close(fig6b);
        close(fig6c);
        close(fig6d);
        close(fig6e);
        close(fig6f);
        for i=1:length(fig6g)
            close(fig6g(i));
        end
%    end
%end
%quit
