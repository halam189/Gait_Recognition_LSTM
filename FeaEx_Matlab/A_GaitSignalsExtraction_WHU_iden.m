clear all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%	History: USED FOR PROCESSING THE WHU DATASET - GAIT RECOGNITION IN THE
%	WILD
%		using the dataset #3 in the paper
%           + concatenate the segments of origninal data and remove the overlapping
%           + the concatenation is processed in timing order
%           + split the concatenated data into segments which overlaps each other 95%
%           + divide the training into training and validating datasets
%               + training
%               + validating
%           + all segments from testing sets are used for testing (no overlapping)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%dataPath = '_rawdata\\OU_data/';
% addpath('funcProcessingExtract');	%folder contains main functions for processing raw gait data

clear all;
dataPath = '..\\Dataset\\raw_data\\WHU_data\\';

rOverlap = 0.97;            % overlap 
data_length = 80;           % length of each segment
validating_number = 2;      % number of segments to be used for validating
nJump = floor(data_length * (1-rOverlap));

% read the segmented data and concatenated to fit our model
train_acc_x = dlmread(strcat(dataPath,'train_acc_x.txt'),' ', 0, 0);
train_acc_y = dlmread(strcat(dataPath,'train_acc_y.txt'),' ', 0, 0);
train_acc_z = dlmread(strcat(dataPath,'train_acc_z.txt'),' ', 0, 0);

train_gyr_x = dlmread(strcat(dataPath,'train_gyr_x.txt'),' ', 0, 0);
train_gyr_y = dlmread(strcat(dataPath,'train_gyr_y.txt'),' ', 0, 0);
train_gyr_z = dlmread(strcat(dataPath,'train_gyr_z.txt'),' ', 0, 0);
train_label = dlmread(strcat(dataPath,'y_train.txt'),' ', 0, 0);

test_acc_x = dlmread(strcat(dataPath,'test_acc_x.txt'),' ', 0, 0);
test_acc_y = dlmread(strcat(dataPath,'test_acc_y.txt'),' ', 0, 0);
test_acc_z = dlmread(strcat(dataPath,'test_acc_z.txt'),' ', 0, 0);

test_gyr_x = dlmread(strcat(dataPath,'test_gyr_x.txt'),' ', 0, 0);
test_gyr_y = dlmread(strcat(dataPath,'test_gyr_y.txt'),' ', 0, 0);
test_gyr_z = dlmread(strcat(dataPath,'test_gyr_z.txt'),' ', 0, 0);
test_label = dlmread(strcat(dataPath,'y_test.txt'),' ', 0, 0);

% get the number of users
user_ID = unique(train_label);
user_number = size(user_ID, 1);

% concatenate the data for each user

RawData = cell(user_number, 5);
for iUser = 1: user_number
    % training data
    % get the index of all samples belong to the current user
    idxes = find(train_label == iUser);
    
    % get all training data of current user 
    cur_user_training_acc_x = train_acc_x(idxes, :);
    cur_user_training_acc_y = train_acc_y(idxes, :);
    cur_user_training_acc_z = train_acc_z(idxes, :);
    
    cur_user_training_gyr_x = train_gyr_x(idxes, :);
    cur_user_training_gyr_y = train_gyr_y(idxes, :);
    cur_user_training_gyr_z = train_gyr_z(idxes, :);
    
    cur_user_data = func_DataReArrange(cur_user_training_acc_x, cur_user_training_acc_y, cur_user_training_acc_z, cur_user_training_gyr_x , cur_user_training_gyr_y, cur_user_training_gyr_z);
    RawData{iUser, 1} = cur_user_data;
    
    
    % testing data
    % get the index of all samples belong to the current user
    idxes = find(test_label == iUser);
    
    % get all training data of current user 
    cur_user_testing_acc_x = test_acc_x(idxes, :);
    cur_user_testing_acc_y = test_acc_y(idxes, :);
    cur_user_testing_acc_z = test_acc_z(idxes, :);
    
    cur_user_testing_gyr_x = test_gyr_x(idxes, :);
    cur_user_testing_gyr_y = test_gyr_y(idxes, :);
    cur_user_testing_gyr_z = test_gyr_z(idxes, :);
    
    cur_user_data = func_DataReArrange(cur_user_testing_acc_x, cur_user_testing_acc_y, cur_user_testing_acc_z, cur_user_testing_gyr_x , cur_user_testing_gyr_y, cur_user_testing_gyr_z);
    RawData{iUser, 2} = cur_user_data;
end

% segment data to fixed-length 

min_number_temp = 0;
for iUser = 1: length(RawData)
    % training and validating datasets
    cur_user_training = RawData{iUser, 1};
    cur_user_training = cur_user_training';
    mSegmentedData = [];
    
	iBegin = 1;
    iEnd = iBegin -1 + data_length;
    while (iEnd <= size(cur_user_training, 1))
        mSegment = cur_user_training(iBegin:iEnd, :);

        mSegment = [mSegment(:, 1); mSegment(:, 2); mSegment(:, 3); mSegment(:, 4); mSegment(:, 5); mSegment(:, 6); iUser];
        mSegmentedData = [mSegmentedData; mSegment'];

        iBegin = iBegin + nJump - 1;
        iEnd = iBegin -1 + data_length;
    end
    RawData{iUser, 3} = mSegmentedData; 
        
    if min_number_temp == 0 || min_number_temp > size(mSegmentedData,1)
        min_number_temp = size(mSegmentedData, 1);
    end
    
    % testing dataset
    cur_user_testing = RawData{iUser, 2};
    cur_user_testing = cur_user_testing';
    mSegmentedData = [];
    
	iBegin = 1;
    iEnd = iBegin -1 + data_length;
    while (iEnd <= size(cur_user_testing, 1))
        mSegment = cur_user_testing(iBegin:iEnd, :);

        mSegment = [mSegment(:, 1); mSegment(:, 2); mSegment(:, 3); mSegment(:, 4); mSegment(:, 5); mSegment(:, 6); iUser];
        mSegmentedData = [mSegmentedData; mSegment'];

        iBegin = iBegin + nJump - 1;
        iEnd = iBegin -1 + data_length; % no overlapping for testing data
    end
    RawData{iUser, 4} = mSegmentedData; 
end

min_number_temp

training_number = min_number_temp - validating_number;

training_data = [];
testing_data = [];
validating_data = [];

for iSession = 1: length(RawData)
    cur_session_train_data = RawData{iSession, 3};
    % shuffle the data template
    vnRandArr = randperm(size(cur_session_train_data, 1), size(cur_session_train_data, 1));
    cur_session_train_data = cur_session_train_data(vnRandArr, :);

    % training and validating
    cur_training_data = cur_session_train_data(1:training_number, :);
    cur_session_train_data(1:training_number, :) = [];
    cur_validating_data = cur_session_train_data(1:validating_number, :);      

    training_data = [training_data; cur_training_data];
    validating_data = [validating_data; cur_validating_data];

    % testing
    cur_session_test_data = RawData{iSession, 4};
    testing_data = [testing_data; cur_session_test_data];

end

% save to file
str_folder_path = '..\\Dataset\\segments\\';
str_file_name = strcat('WHU_equal_',...
    num2str(length(user_ID)),...
    '_trn', num2str(training_number),...
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

% validation data 
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
