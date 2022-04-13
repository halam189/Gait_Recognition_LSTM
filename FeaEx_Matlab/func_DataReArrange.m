function [cur_user_data] = func_DataReArrange(cur_user_training_acc_x, cur_user_training_acc_y, cur_user_training_acc_z, cur_user_training_gyr_x, cur_user_training_gyr_y, cur_user_training_gyr_z)
    nov_for_matching = 5;    
    % find the order in time of the templates
    timing_order = [1];
    % select the first template
    found = 1;
    cur_template_idx = 1;
    cur_template = cur_user_training_acc_x(cur_template_idx, :);
    
    % find forward
    while found==1
        found = 0;
        % get nov_for_matching last elements of cur_template
        matching_values = cur_template(128-nov_for_matching+1:128);
        for iTemplate = 1:size(cur_user_training_acc_x, 1)
            template_ith = cur_user_training_acc_x(iTemplate, :);
            % matching
            difference = false;
            
            for iValue = 1: nov_for_matching
                if matching_values(iValue) ~= template_ith(64-nov_for_matching+iValue)
                    difference = true;
                    break;
                end
            end
            if difference == false
                found = 1;
                break;
            end
        end
        if found == 1
            timing_order = [timing_order iTemplate];
            cur_template = cur_user_training_acc_x(iTemplate, :);
        end
    end
    % find backward
    found = 1;
    cur_template_idx = 1;
    cur_template = cur_user_training_acc_x(cur_template_idx, :);
    
    while found==1
        found = 0;
        % get nov_for_matching last elements of cur_template
        matching_values = cur_template(64-nov_for_matching+1:64);
        for iTemplate = 1:size(cur_user_training_acc_x, 1)
            template_ith = cur_user_training_acc_x(iTemplate, :);
            % matching
            difference = false;
            
            for iValue = 1: nov_for_matching
                if matching_values(iValue) ~= template_ith(128-nov_for_matching+iValue)
                    difference = true;
                end
            end
            if difference == false
                found = 1;
                break;
            end
        end
        if found == 1
            timing_order = [iTemplate timing_order];
            cur_template = cur_user_training_acc_x(iTemplate, :);
        end
    end
    
    % re-arrange the data
    cur_user_training_acc_x = cur_user_training_acc_x(timing_order, :);
    cur_user_training_acc_y = cur_user_training_acc_y(timing_order, :);
    cur_user_training_acc_z = cur_user_training_acc_z(timing_order, :);
    
    cur_user_training_gyr_x = cur_user_training_gyr_x(timing_order, :);
    cur_user_training_gyr_y = cur_user_training_gyr_y(timing_order, :);
    cur_user_training_gyr_z = cur_user_training_gyr_z(timing_order, :);
    
    % concatenate the data
    % acc_x
    cur_user_acc_x = cur_user_training_acc_x(1, :);
    for i=2: size(cur_user_training_acc_x,1)
        cur_template = cur_user_training_acc_x(i, :);
        cur_template(1: 64) = [];
        cur_user_acc_x = [cur_user_acc_x cur_template];
    end
    cur_user_data = cur_user_acc_x;
    % acc_y
    cur_user_acc_y = cur_user_training_acc_y(1, :);
    for i=2: size(cur_user_training_acc_y,1)
        cur_template = cur_user_training_acc_y(i, :);
        cur_template(1: 64) = [];
        cur_user_acc_y = [cur_user_acc_y cur_template];
    end
    cur_user_data = [cur_user_data; cur_user_acc_y];
    % acc_Z
    cur_user_acc_z = cur_user_training_acc_z(1, :);
    for i=2: size(cur_user_training_acc_z,1)
        cur_template = cur_user_training_acc_z(i, :);
        cur_template(1: 64) = [];
        cur_user_acc_z = [cur_user_acc_z cur_template];
    end
    cur_user_data = [cur_user_data; cur_user_acc_z];
    % gyr_x
    cur_user_gyr_x = cur_user_training_gyr_x(1, :);
    for i=2: size(cur_user_training_gyr_x,1)
        cur_template = cur_user_training_gyr_x(i, :);
        cur_template(1: 64) = [];
        cur_user_gyr_x = [cur_user_gyr_x cur_template];
    end
    cur_user_data = [cur_user_data; cur_user_gyr_x];
    
    % gyr_y 
    cur_user_gyr_y = cur_user_training_gyr_y(1, :);
    for i=2: size(cur_user_training_gyr_y, 1)
        cur_template = cur_user_training_gyr_y(i, :);
        cur_template(1: 64) = [];
        cur_user_gyr_y = [cur_user_gyr_y cur_template];
    end
    cur_user_data = [cur_user_data; cur_user_gyr_y];
    
    % gyr_z
    cur_user_gyr_z = cur_user_training_gyr_z(1, :);
    for i=2: size(cur_user_training_gyr_z,1)
        cur_template = cur_user_training_gyr_z(i, :);
        cur_template(1: 64) = [];
        cur_user_gyr_z = [cur_user_gyr_z cur_template];
    end
    cur_user_data = [cur_user_data; cur_user_gyr_z];
end