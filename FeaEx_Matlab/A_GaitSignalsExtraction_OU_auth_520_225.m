%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%		Extract from raw gait data (OU-ISIR dataset) the fixed-length
%		segments and divide into training/testing/validating sets to be
%		used for user authentication task
%           data of 520 users are used for training and validating
%           data of 224 users are used for testing
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;

nTraining_user = 520; 
rOverlap = 0.97;
data_length = 100;

nJump = floor(data_length * (1-rOverlap));

dataPath = '..\\Dataset\\raw_data\\OU_ISIR\\';
dataUsage = 'Center_seq';
fData = dir(fullfile(dataPath,strcat('*',dataUsage,'*.csv')));
RawData = cell(length(fData),12);
for i=1:length(fData)
    curAccelerationData = dlmread(strcat(dataPath,fData(i).name),',',2,0);
    
    idxID = strfind(fData(i).name,'_ID');
    curID = str2double(fData(i).name(idxID+3:idxID+8));
    idxSessionID = strfind(fData(i).name,'_seq');
    curSessionID = str2double(fData(i).name(idxSessionID+4));
    RawData{i,1} = curAccelerationData;
    RawData{i,2} = idxSessionID;
    RawData{i,3} = curSessionID;
    RawData{i,4} = curID;
end

%get list UserID (stored in vnUserID)
user_ID= [];
nRawDataLen = length(RawData);
start = 1;
iend = nRawDataLen;

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

% divide into training and testing datasets

training_data_all = cell(nTraining_user, 4);

vnRandArr = randperm(user_number, user_number);
rand_user_ID = vnRandArr;

% extracting training data 
for i=1:nTraining_user
    % get user ID
    user_ID = rand_user_ID(i);
    mrCurUserSegment = [];
    for iFind = 1:nRawDataLen
        if user_ID == RawData{iFind, 2}
            % get data
            mrCurSequence = RawData{iFind, 1};
            iBegin = 1;
            iEnd = iBegin -1 + data_length;
            while (iEnd <= size(mrCurSequence, 1))
                mSegment = mrCurSequence(iBegin:iEnd, :);

                mSegment = [mSegment(:, 1); mSegment(:, 2); mSegment(:, 3); mSegment(:, 4); mSegment(:, 5); mSegment(:, 6); i-1];
                mrCurUserSegment = [mrCurUserSegment; mSegment'];

                iBegin = iBegin + nJump - 1;
                iEnd = iBegin -1 + data_length;
            end
        end
    end
    training_data_all{i,1} = mrCurUserSegment;
end

training_data = [];
validating_data = [];
testing_data = [];

% min_template_num = 0;
% for i=1:nTraining_user
%     mrData = training_data_all{i,1};
%     training_data_all{i,2} = size(mrData,1);
%     if min_template_num ==0 || min_template_num >size(mrData,1)
%         min_template_num = size(mrData,1) ;
%     end
% end
% min_template_num
for i=1:nTraining_user
    mrData = training_data_all{i,1};
    training_data = [training_data; mrData(1:size(mrData,1)-4, :)];
    validating_data = [validating_data; mrData(size(mrData,1)-3:size(mrData,1), :)];
end

% extracting training data 
for i=nTraining_user+1:user_number
    % get user ID
    user_ID = rand_user_ID(i);
    for iFind = 1:nRawDataLen
        if user_ID == RawData{iFind, 2}
            % extract this 
            % get data
            mrCurSequence = RawData{iFind, 1};
            iBegin = 1;
            iEnd = iBegin -1 + data_length;
            while (iEnd <= size(mrCurSequence, 1))
                mSegment = mrCurSequence(iBegin:iEnd, :);

                mSegment = [mSegment(:, 1); mSegment(:, 2); mSegment(:, 3); mSegment(:, 4); mSegment(:, 5); mSegment(:, 6); i-nTraining_user-1];
                testing_data = [testing_data; mSegment'];

                iBegin = iBegin + data_length - 1;
                iEnd = iBegin -1 + data_length;
            end
        end
    end
end


% save to file
str_folder_path = '..\\Dataset\\segments\\';
str_file_name = strcat('_auth_1_OU_520_225',...
    num2str(length(user_ID)),...
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
    fprintf(fileID,'%.8f ',vr_signals); 

    % writing the user ID
    nUserID = vr_data_row(end);
	fprintf(fileID,'%d\n',nUserID); 
end	

fclose(fileID);

% Testing Data File    
fileID = fopen(str_file_path_test,'w');
for(iRow = 1:(size(testing_data,1)))
    % write the signals
    vr_data_row = testing_data(iRow,:);
    vr_signals = vr_data_row(1: data_length*6);
    fprintf(fileID,'%.8f ',vr_signals); 
    
    % writing the user ID
    nUserID = vr_data_row(end);
	fprintf(fileID,'%d\n',nUserID); 
end	
fclose(fileID);

% validation data 
fileID = fopen(str_file_path_valida,'w');
for(iRow = 1:(size(validating_data,1)))
    % write the signals
    vr_data_row = validating_data(iRow,:);
    vr_signals = vr_data_row(1: data_length*6);
    fprintf(fileID,'%.8f ',vr_signals); 
    
    % writing the user ID
    nUserID = vr_data_row(end);
	fprintf(fileID,'%d\n',nUserID); 
end	
fclose(fileID);
