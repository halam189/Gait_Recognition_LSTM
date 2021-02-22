%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%		Extract from raw gait data (OU-ISIR dataset) the fixed-length
%		segments and divide into training/testing/validating sets to be
%		used for identification task
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
% addpath('funcProcessingExtract');	%folder contains main functions for processing raw gait data

% PARAMETERS
data_length = 100;              % length of the segment (100 as specified in the paper)
rOverlap = 0.97;                % overlapping between two consecutive segments
validation_rate = 0.2;          % amount of data to be used as validation set
nJump = floor(data_length * (1-rOverlap));

%
% 0. Load data of accelerometer and gyroscope data
%
dataPath = '..\\Dataset\\raw_data\\OU_ISIR\\';        % path of the raw dataset
dataUsage = 'Center_seq';
fData = dir(fullfile(dataPath,strcat('*',dataUsage,'*.csv')));
% the main data structure for containing all dataset and its information
RawData = cell(length(fData),12);

for i=1:length(fData)
    % read file
    curAccelerationData = dlmread(strcat(dataPath,fData(i).name),',',2,0);
    
    % parse the data information 
    idxID = strfind(fData(i).name,'_ID');
    curID = str2double(fData(i).name(idxID+3:idxID+8));
    idxSessionID = strfind(fData(i).name,'_seq');
    curSessionID = str2double(fData(i).name(idxSessionID+4));
    
    % add to main data structure
    GyroscopeData = curAccelerationData(:, 1:3);
    RawData{i,1} = curAccelerationData;
    RawData{i,2} = idxSessionID;
    RawData{i,3} = curSessionID;
    RawData{i,4} = curID; 
end


nRawDataLen = length(RawData);
start = 1;
iend = nRawDataLen;
%
% 0.1. get list of user ID (stored in vnUserID)
%
user_ID= [];

for(iData = start: iend)
    user_ID = [user_ID RawData{iData,4}];
end
user_ID = unique(user_ID);

%find the number of user
user_number = size(user_ID, 2);

curNewUserID = 1;
preOldID = RawData{start,4};
RawData{start,2} = curNewUserID;
for(iData = (start+1): iend)
    curOldID = RawData{iData,4};
    if(curOldID > preOldID)
        curNewUserID = curNewUserID+1;
        preOldID = curOldID;
    end
    RawData{iData,2} = curNewUserID; 
end

%
% 1.1 divide into training and testing datasets
%
training_data = [];
testing_data = [];

for iSession = 1: length(RawData)
    curSessionData = RawData{iSession, 1};
    curUserID = RawData{iSession, 2};
    number_of_tr_signal = floor(size(curSessionData,1)*0.5);

    % training
    mTrainingSegment = curSessionData(1:number_of_tr_signal,:);
    mSegmentedData = [];
    iBegin = 1;
    iEnd = iBegin -1 + data_length;
    while (iEnd <= size(mTrainingSegment, 1))
        mSegment = mTrainingSegment(iBegin:iEnd, :);

        mSegment = [mSegment(:, 1); mSegment(:, 2); mSegment(:, 3); mSegment(:, 4); mSegment(:, 5); mSegment(:, 6); curUserID];
        mSegmentedData = [mSegmentedData; mSegment'];

        iBegin = iBegin + nJump - 1;
        iEnd = iBegin -1 + data_length;
    end
    training_data = [training_data; mSegmentedData];

    % testing 
    mTestingSegment = curSessionData(number_of_tr_signal+1:size(curSessionData,1),:);
    mSegmentedData = [];
    iBegin = 1;
    iEnd = iBegin -1 + data_length;
    while (iEnd <= size(mTestingSegment, 1))
        mSegment = mTestingSegment(iBegin:iEnd, :);

        mSegment = [mSegment(:, 1); mSegment(:, 2); mSegment(:, 3); mSegment(:, 4); mSegment(:, 5); mSegment(:, 6); curUserID];
        mSegmentedData = [mSegmentedData; mSegment'];

        iBegin = iBegin + data_length - 1;
        iEnd = iBegin -1 + data_length;
    end
    testing_data = [testing_data; mSegmentedData];

end

%
% 1.2. dividing data to testing and validating
%
validating_num = floor(size(testing_data, 1)*validation_rate);

vnRandArr = randperm(size(testing_data, 1), size(testing_data, 1));
testing_data = testing_data(vnRandArr, :);

validating_data = testing_data(1:validating_num, :);
testing_data(1:validating_num, :) =[];
%
% Save to file
%

% file path that the extracted segments will be saved
str_folder_path = '..\\Dataset\\segments\\';
str_file_name = strcat('OU_segments_len_',...
    num2str(data_length),...
    '_ovl', num2str(rOverlap*10)...
    )

str_file_path_train = strcat(str_folder_path,str_file_name,'_Train');
str_file_path_test = strcat(str_folder_path,str_file_name,'_Test');
str_file_path_valida = strcat(str_folder_path,str_file_name,'_Vali');

% Training Data File
fileID = fopen(str_file_path_train,'w');

for(iRow = 1:(size(training_data,1)))
    % write the signals
    vr_data_row = training_data(iRow,:);
    vr_signals = vr_data_row(1: data_length*6);
    fprintf(fileID,'%.10f ',vr_signals); 

    % writing the user ID
    nUserID = vr_data_row(end);
	fprintf(fileID,'%d\n',nUserID-1); 
end	

fclose(fileID);

% Testing Data File    
fileID = fopen(str_file_path_test,'w');
for(iRow = 1:(size(testing_data,1)))
    % write the signals
    vr_data_row = testing_data(iRow,:);
    vr_signals = vr_data_row(1: data_length*6);
    fprintf(fileID,'%.10f ',vr_signals); 
    
    % writing the user ID
    nUserID = vr_data_row(end);
	fprintf(fileID,'%d\n',nUserID-1); 
end	
fclose(fileID);

% validation data file
fileID = fopen(str_file_path_valida,'w');
for(iRow = 1:(size(validating_data,1)))
    % write the signals
    vr_data_row = validating_data(iRow,:);
    vr_signals = vr_data_row(1: data_length*6);
    fprintf(fileID,'%.10f ',vr_signals); 
    
    % writing the user ID
    nUserID = vr_data_row(end);
	fprintf(fileID,'%d\n',nUserID-1); 
end	
fclose(fileID);
