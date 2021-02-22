%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%	Program Descriptions - Plotting Figure 1 in the paper 
%
%		This script plot the accelerometor and gyroscope signals and 
% illustrate the fixed length segmentation 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
clear;

%load a random raw gait data file
mrGaitData = dlmread('..\Dataset\raw_data\\OU_ISIR\\T0_ID153748_Center_seq0.csv',',',2,0);

% generate the timestamps of gait signals 
vrTimeStamp = linspace (1, size(mrGaitData,1), size(mrGaitData,1));

s = 1;  % starting point of the gait sequence
e = 750;    % ending point
offset = 30;
time = [1 100 200 300 400 500];
dataaa = mrGaitData(time, 1);

%plotting ACCELERATION figure
figure(1);
hold on

	%plotting X
	subplot(3,1,1) ; 
    plot(vrTimeStamp(s:e), mrGaitData(s+offset:e+offset,1),'Color','b','LineWidth',2);
    hold on
    stem(time,dataaa,'MarkerSize',5,'MarkerEdgeColor','None', 'LineWidth',1.5,'Color','None');
    % plot the segment's boundaries 
    for iline = 1 : length(time)
        xline(time(iline),'r-','LineWidth',2)
    end
	xlabel('Sample','FontSize',12);
	ylabel('Acceleration (g)','FontSize',12);
	title ('X axis','FontSize',12);
    
	%plotting Y
    set(gca,'FontSize',12);
    set(gca,'xlim',[1 550]);
	subplot(3,1,2) ; 
    plot(vrTimeStamp(s:e), mrGaitData(s+offset:e+offset,2),'Color','b','LineWidth',2);
    hold on
    % plot the segment's boundaries 
    for iline = 1 : length(time)
        xline(time(iline),'r-','LineWidth',2)
    end
    stem(time,dataaa,'MarkerSize',5,'MarkerEdgeColor','None', 'LineWidth',1.5,'Color','None');
    title ('Y axis','FontSize',12);
	xlabel('Sample','FontSize',12);
	ylabel('Acceleration (g)','FontSize',12);
    
	%plotting Z
    set(gca,'FontSize',12);
    set(gca,'xlim',[1 550]);
    
    subplot(3,1,3) ;
    plot(vrTimeStamp(s:e), mrGaitData(s+offset:e+offset,3),'Color','b','LineWidth',2);
    % plot the segment's boundaries 
    for iline = 1 : length(time)
        xline(time(iline),'r-','LineWidth',2)
    end
    
    hold on
    stem(time,dataaa,'MarkerSize',5,'MarkerEdgeColor','None', 'LineWidth',1.5,'Color','None');
    title ('Z axis','FontSize',12);
	xlabel('Sample','FontSize',12);
	ylabel('Acceleration (g)','FontSize',12);
	set(gca,'FontSize',12);
    set(gca,'xlim',[1 550]);
x0=150;
y0=150;
width=1080;
height=830;
set(gcf,'position',[x0,y0,width,height])

% PLOTTINGYROSCOPE   
figure(2);
hold on
	%plotting X
	subplot(3,1,1) ; 
    plot(vrTimeStamp(s:e), mrGaitData(s+offset:e+offset,4),'Color','b','LineWidth',2);
    hold on
    stem(time,dataaa,'MarkerSize',5,'MarkerEdgeColor','None', 'LineWidth',1.5,'Color','None');
    for iline = 1 : length(time)
        xline(time(iline),'r-','LineWidth',2)
    end
	xlabel('Sample','FontSize',12);
	ylabel('Angular velocity (rad/s)','FontSize',12);
	title ('X axis','FontSize',12);

    %plotting Y
    set(gca,'FontSize',12);
    set(gca,'xlim',[1 550]);
	subplot(3,1,2) ; 
    plot(vrTimeStamp(s:e), mrGaitData(s+offset:e+offset,5),'Color','b','LineWidth',2);
    hold on
    for iline = 1 : length(time)
        xline(time(iline),'r-','LineWidth',2)
    end
    %stem(time,dataaa,'MarkerSize',5,'MarkerEdgeColor','None', 'LineWidth',2,'Color','None');
    title ('Y axis','FontSize',12);
	xlabel('Sample','FontSize',12);
	ylabel('Angular velocity (rad/s)','FontSize',12);
	
    %plotting Z
    set(gca,'FontSize',12);
    set(gca,'xlim',[1 550]);
    
    subplot(3,1,3) ;
    plot(vrTimeStamp(s:e), mrGaitData(s+offset:e+offset,6),'Color','b','LineWidth',2);
    for iline = 1 : length(time)
        xline(time(iline),'r-','LineWidth',2)
    end
    
    hold on
    stem(time,dataaa,'MarkerSize',5,'MarkerEdgeColor','None', 'LineWidth',1.5,'Color','None');
    title ('Z axis','FontSize',12);
	xlabel('Sample','FontSize',12);
	ylabel('Angular velocity (rad/s)','FontSize',12);
	set(gca,'FontSize',12);
    set(gca,'xlim',[1 550]);
x0=650;
y0=150;
width=1080;
height=830;
set(gcf,'position',[x0,y0,width,height])