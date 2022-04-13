%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%	History: USED FOR PROCESSING THE WHU DATASET - GAIT RECOGNITION IN THE
%	WILD
%		using the dataset #3 in the paper
%           + concatenate the segments of origninal data and remove the overlapping
%           + the concatenation is processed in timing order
%           + split the concatenated data into segments which overlaps each other 95%
%           + divide the training into training and validating datasets
%           + all segments from testing sets are used for testing (no overlapping)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%dataPath = '_rawdata\\OU_data/';

% 98 - 20


clear all;
dataPath = '..\\Dataset\\raw_data\\WHU_data\\';

rOverlap = 0.94;
data_length = 80;
nJump = floor(data_length * (1-rOverlap));


n_training_users = 98;
n_testing_users = 20;

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

data_size_all = [];
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
    
    train_data_size1 = size(cur_user_data, 2);
    
    % testing data
    % get the index of all samples belong to the current user
    idxes = find(test_label == iUser);
    % get all testing data of current user 
    cur_user_testing_acc_x = test_acc_x(idxes, :);
    cur_user_testing_acc_y = test_acc_y(idxes, :);
    cur_user_testing_acc_z = test_acc_z(idxes, :);
    
    cur_user_testing_gyr_x = test_gyr_x(idxes, :);
    cur_user_testing_gyr_y = test_gyr_y(idxes, :);
    cur_user_testing_gyr_z = test_gyr_z(idxes, :);
    
    cur_user_data = func_DataReArrange(cur_user_testing_acc_x, cur_user_testing_acc_y, cur_user_testing_acc_z, cur_user_testing_gyr_x , cur_user_testing_gyr_y, cur_user_testing_gyr_z);
    RawData{iUser, 2} = cur_user_data;
    
    % train_data_size2 = size(cur_user_data, 2);
    data_size_all = [data_size_all train_data_size1];
end

[data_size_all, index_datasize] = sort(data_size_all, 'descend');

% randomly select 98 users for training
% vnRandArr = randperm(user_number, user_number);
rand_user_ID = user_ID(index_datasize);

% segment data to fixed-length 
n_training_users = 98;
n_testing_users = 20;

%
mrTrainingData = []; %data of 98 users
mrValidatingData = []; %data of 98 users
mrTestingData = [];     % data of 20 users

min_data_size = data_size_all(98);

for iUser = 1: n_training_users
    mrCurUserData = [];
    iUserID = rand_user_ID(iUser);
    
    mrCurUserData = RawData{iUserID, 1};
    mrCurUserData(:, min_data_size:end)= [];
    RawData{iUserID, 1} = mrCurUserData;
    
end

% split data sequence to segments

for iUser = 1: n_training_users
    mrCurUserData_train = [];
    mrCurUserData_vali = [];
    iUserID = rand_user_ID(iUser);
    
    %get data of this user
    cur_user_data = RawData{iUserID, 1};
    cur_user_data = cur_user_data';
    iBegin = 1;
    iEnd = iBegin -1 + data_length;
    while (iEnd <= size(cur_user_data, 1))
        mSegment = cur_user_data(iBegin:iEnd, :);

        mSegment = [mSegment(:, 1); mSegment(:, 2); mSegment(:, 3); mSegment(:, 4); mSegment(:, 5); mSegment(:, 6); iUser];
        mrCurUserData_train = [mrCurUserData_train; mSegment'];

        iBegin = iBegin + nJump - 1;
        iEnd = iBegin -1 + data_length;
    end
    
    cur_user_data = RawData{iUserID, 2};
    cur_user_data = cur_user_data';
    iBegin = 1;
    iEnd = iBegin -1 + data_length;
    while (iEnd <= size(cur_user_data, 1))
        mSegment = cur_user_data(iBegin:iEnd, :);

        mSegment = [mSegment(:, 1); mSegment(:, 2); mSegment(:, 3); mSegment(:, 4); mSegment(:, 5); mSegment(:, 6); iUser];
        mrCurUserData_vali = [mrCurUserData_vali; mSegment'];

        iBegin = iBegin + data_length - 1;
        iEnd = iBegin -1 + data_length;
    end
    RawData{iUserID, 3} = mrCurUserData_train;
    RawData{iUserID, 4} = mrCurUserData_vali;
    
    mrTrainingData = [mrTrainingData; mrCurUserData_train];
    mrValidatingData = [mrValidatingData; mrCurUserData_vali];
    
end
% testing dataa
for iUser = n_training_users+1: user_number
    iUser
    mrCurUserData = [];
    iUserID = rand_user_ID(iUser);
    
    %get data of this user
    cur_user_data = RawData{iUserID, 1};
    cur_user_data = cur_user_data';
    iBegin = 1;
    iEnd = iBegin -1 + data_length;
    while (iEnd <= size(cur_user_data, 1))
        mSegment = cur_user_data(iBegin:iEnd, :);

        mSegment = [mSegment(:, 1); mSegment(:, 2); mSegment(:, 3); mSegment(:, 4); mSegment(:, 5); mSegment(:, 6); iUser-n_training_users];
        mrCurUserData = [mrCurUserData; mSegment'];

        iBegin = iBegin + data_length - 1;
        iEnd = iBegin -1 + data_length;
    end
    
    cur_user_data = RawData{iUserID, 2};
    cur_user_data = cur_user_data';
    iBegin = 1;
    iEnd = iBegin -1 + data_length;
    while (iEnd <= size(cur_user_data, 1))
        mSegment = cur_user_data(iBegin:iEnd, :);

        mSegment = [mSegment(:, 1); mSegment(:, 2); mSegment(:, 3); mSegment(:, 4); mSegment(:, 5); mSegment(:, 6); iUser-n_training_users];
        mrCurUserData = [mrCurUserData; mSegment'];

        iBegin = iBegin + data_length - 1;
        iEnd = iBegin -1 + data_length;
    end
    mrTestingData = [mrTestingData; mrCurUserData];
end

% save to file
str_folder_path = '..\\Dataset\\segments\\';
str_file_name = strcat('WHU80_authen_equal_98_20',...
    num2str(length(user_ID)),...
    '_ovl', num2str(rOverlap*10)...
    )

str_file_path_train = strcat(str_folder_path,str_file_name,'_Train');
str_file_path_test = strcat(str_folder_path,str_file_name,'_Test');
str_file_path_vali = strcat(str_folder_path,str_file_name,'_Vali');

% Training Data File
fileID = fopen(str_file_path_train,'w');

for iRow = 1:(size(mrTrainingData,1))
    % write the signals
    vr_data_row = mrTrainingData(iRow,:);
    vr_signals = vr_data_row(1: data_length*6);
    fprintf(fileID,'%.10f ',vr_signals); 

    % writing the user ID
    nUserID = vr_data_row(end);
	fprintf(fileID,'%d\n',nUserID-1); 
end	

fclose(fileID);


% Validating Data File
fileID = fopen(str_file_path_vali,'w');

for(iRow = 1:(size(mrValidatingData,1)))
    % write the signals
    vr_data_row = mrValidatingData(iRow,:);
    vr_signals = vr_data_row(1: data_length*6);
    fprintf(fileID,'%.10f ',vr_signals); 

    % writing the user ID
    nUserID = vr_data_row(end);
	fprintf(fileID,'%d\n',nUserID-1); 
end	
fclose(fileID);

% Testing Data File    
fileID = fopen(str_file_path_test,'w');
for(iRow = 1:(size(mrTestingData,1)))
    % write the signals
    vr_data_row = mrTestingData(iRow,:);
    vr_signals = vr_data_row(1: data_length*6);
    fprintf(fileID,'%.10f ',vr_signals); 
    
    % writing the user ID
    nUserID = vr_data_row(end);
	fprintf(fileID,'%d\n',nUserID-1); 
end	
fclose(fileID);
