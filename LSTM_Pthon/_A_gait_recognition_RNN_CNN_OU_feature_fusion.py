import torch
import torch.tensor
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from authen_netw import *

import torch.nn.functional as F
from GaitDataset import *


# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


############  TRAINING SETTINGS  #################

result = np.array([])
experiment_times = 1        # number of experiments
validation_threshold_RNN = 15  # after #validation_threshold times that the validation loss does not decrease, the training process stops
validation_threshold_CNN = 15
validation_threshold_Fuse = 15

learning_rate_CNN = 0.00015
learning_rate_RNN = 0.15
learning_rate_Fuse = 0.00001

using_CNN = 1
using_RNN = 1

###########  NETWORK PARAMETERS  #################

data_length = 100       # number of signals per each gait cycle
dimension = 6           # number of channnels

rnn_input_dim = 10     # the number of scalars that is fed to an LSTM node at each stage           #^#
rnn_hidden_dim = 40   # the number of scalars that is returned from LSTM node at each stage       #^#
rnn_layer_dim = 2      # the number of layers in the RNN network                                    #^#

score_fusion = 1
number_of_template = 5
fusion_all_case = 1
##################  DATASET  ######################

file_name = "..\\Dataset\\segments\\A_OU_segments_len_100_ovl9.7_"

user_number = 745
rnn_output_dim = user_number

training_file = file_name + "train"
testing_file = file_name + "test"
validating_file = file_name + "vali"


bLoadTrainedModel_RNN = True   # load a trained LSTM model
trained_model_path_RNN = "..\\Dataset\\trained_models\\_A_gait_recognition_RNN_OU"

bLoadTrainedModel_CNN = True   # load a trained CNN model
trained_model_path_CNN = "..\\Dataset\\trained_models\\_A_gait_recognition_CNN_OU"

bLoadTrainedModel_FC = True   # load a trained FC model
trained_model_path_FC = "..\\Dataset\\trained_models\\_A_gait_recognition_FC_OU"

if (not bLoadTrainedModel_RNN) or (not bLoadTrainedModel_CNN) or (not bLoadTrainedModel_FC):
    training_dataset = GaitSessionDataset(file_path=training_file, data_length=data_length, dimension=dimension)
    validation_dataset = GaitSessionDataset(file_path=validating_file, data_length=data_length, dimension=dimension)

    train_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)


# # # # # # # # # # # # # # # #  TRAINING  # # # # # # # # # # # # # # # # # # # # # # #
for iTimes in range(0, experiment_times):
    #    ######## TRAIN RNN NETWORK ########
    model_RNN = LSTM_6Chan_OU(rnn_input_dim, rnn_hidden_dim, rnn_layer_dim, rnn_output_dim,
                                   segment_length=data_length)
    model_RNN_store = LSTM_6Chan_OU(rnn_input_dim, rnn_hidden_dim, rnn_layer_dim, rnn_output_dim,
                                         segment_length=data_length)
    model_RNN = model_RNN.to(device)

    if using_RNN:
        optimizer_RNN = torch.optim.SGD(model_RNN.parameters(), lr=learning_rate_RNN, momentum=0.9)
        criterion_rnn = nn.CrossEntropyLoss()
        if bLoadTrainedModel_RNN:
            checkpoint = torch.load(trained_model_path_RNN)
            model_RNN.load_state_dict(checkpoint['model_state_dict'])
            optimizer_RNN.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
        else:
            print('\n---------------------------Training RNN-----------------------------')
            # TRAINING
            validation_count = 0
            previous_loss = 0
            count = 0

            while True:
                count += 1
                model_RNN.train()
                # training
                for batch_idx, data in enumerate(train_dataloader, 0):
                    # zero gradients
                    optimizer_RNN.zero_grad()
                    inputs, label = Variable(data["data"]).float().requires_grad_(), Variable(data["label"]).long()

                    inputs = inputs.to(device)
                    label = label.squeeze()

                    labels = label
                    labels = labels.to(device)

                    # forward pass
                    outputs_RNN, _ = model_RNN(inputs)
                    outputs_RNN = outputs_RNN.to(device)

                    # loss calculation
                    # loss = F.cross_entropy(outputs, labels)
                    loss = F.cross_entropy(outputs_RNN, labels)
                    # loss = F.nll_loss(outputs_RNN, labels)

                    if batch_idx % 100 == 0:
                        print(' {}/{} Train RNN Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            iTimes,
                            experiment_times,
                            count,
                            batch_idx * len(data),
                            len(train_dataloader.dataset),
                            100. * batch_idx / len(train_dataloader),
                            loss.data))

                    loss.backward()
                    optimizer_RNN.step()

                # validating
                curr_validate_loss = 0
                model_RNN.eval()
                for batch_idx, data in enumerate(validation_dataloader, 0):
                    # load test data
                    data, target = Variable(data["data"]).float(), Variable(data["label"]).long()
                    target = target.squeeze()
                    # target = indices_to_one_hot(target, 32)
                    data = data.to(device)
                    # run model
                    with torch.no_grad():
                        output, _ = model_RNN(data)
                        output = output.cpu()

                        if target.dim() != 0 and output.dim() != 0:
                            # get loss and determine predict value
                            curr_validate_loss += F.cross_entropy(output, target).data * data.shape[0]
                            # curr_validate_loss += F.nll_loss(output, target).data*data.shape[0]
                            # curr_validate_loss += criterion_RNN(output, target)

                curr_validate_loss /= len(validation_dataloader.dataset)

                # stop if loss does not reduce for several times
                if previous_loss <= curr_validate_loss and previous_loss != 0:
                    validation_count += 1
                else:
                    validation_count = 0
                    previous_loss = curr_validate_loss
                    model_RNN_store = model_RNN

                print('--- Validation Loss: {:.7f} \t Count: {}'.format(
                    curr_validate_loss.data,
                    validation_count))

                if validation_count == validation_threshold_RNN:
                    model_RNN = model_RNN_store
                    break
            torch.save({'epoch': count,
                    'model_state_dict': model_RNN.state_dict(),
                    'optimizer_state_dict': optimizer_RNN.state_dict(),
                    'loss': previous_loss},
                   trained_model_path_RNN + str(iTimes))

    #    ######## TRAIN CNN NETWORK ########
    model_CNN = CNN_6Chan_OU(usernumber=user_number)
    model_CNN_store = CNN_6Chan_OU(usernumber=user_number)
    model_CNN = model_CNN.to(device)

    #    ######## TRAIN CNN NETWORK ########
    if using_CNN:
        optimizer = torch.optim.Adam(model_CNN.parameters(), lr=learning_rate_CNN)

        if bLoadTrainedModel_CNN:
            checkpoint = torch.load(trained_model_path_CNN)
            model_CNN.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            loss = checkpoint['loss']
        else:
            print('\n---------------------------Training CNN -----------------------------')
            validation_count = 0
            previous_loss = 0
            count = 0
            while True:
                count += 1
                model_CNN.train()
                # training
                for batch_idx, data in enumerate(train_dataloader, 0):
                    # zero gradients
                    optimizer.zero_grad()

                    # get training data
                    inputs, labels = Variable(data["data"]).float(), Variable(data["label"]).long()

                    inputs = inputs.to(device)
                    inputs = inputs.unsqueeze(1)

                    labels = labels.squeeze()
                    labels = labels.to(device)

                    # forward pass
                    outputs, _ = model_CNN(inputs)
                    outputs = outputs.to(device)

                    # loss calculation
                    loss = F.nll_loss(outputs, labels)

                    if batch_idx % 500 == 0:
                        print(' {}/{} Train CNN Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            iTimes,
                            experiment_times,
                            count,
                            batch_idx * len(data),
                            len(train_dataloader.dataset),
                            100. * batch_idx / len(train_dataloader),
                            loss.data))

                    loss.backward()
                    optimizer.step()

                # validating
                curr_validate_loss = 0
                model_CNN.eval()
                for batch_idx, data in enumerate(validation_dataloader, 0):
                    # load test data
                    data, target = Variable(data["data"]).float(), Variable(data["label"]).long()
                    target = target.squeeze()

                    data = data.to(device)
                    data = data.unsqueeze(1)
                    # run model
                    with torch.no_grad():
                        output, _ = model_CNN(data)
                        output = output.cpu()

                        if target.dim() != 0 and output.dim() != 0:
                            # get loss and determine predict value
                            curr_validate_loss += F.nll_loss(output, target, reduction='sum').data

                curr_validate_loss /= len(validation_dataloader.dataset)

                # stop if loss does not reduce for several times
                if previous_loss <= curr_validate_loss and previous_loss != 0:
                    validation_count += 1
                else:
                    validation_count = 0
                    previous_loss = curr_validate_loss
                    model_CNN_store = model_CNN

                print('--- Validation Loss: {:.7f} \t Count: {}'.format(
                    curr_validate_loss.data,
                    validation_count))

                if validation_count == validation_threshold_CNN:
                    model_CNN = model_CNN_store
                    break
            torch.save({'epoch': count,
                'model_state_dict': model_CNN.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': previous_loss},
               trained_model_path_CNN + str(iTimes))

    #    ######## TRAIN THE FUSION LAYER ####
    fusion_input_dim = rnn_hidden_dim*6 + 2520
    model_Fuse = RNN_CNN_Feature_Fusion(input_dim=fusion_input_dim, user_number=user_number)
    model_Fuse_store = RNN_CNN_Feature_Fusion(input_dim=fusion_input_dim, user_number=user_number)
    model_Fuse = model_Fuse.to(device)

    optimizer_fusion = torch.optim.Adam(model_Fuse.parameters(), lr=learning_rate_Fuse)
    train_embedded = np.zeros(0)
    train_label = np.zeros(0)

    if bLoadTrainedModel_FC:
        checkpoint = torch.load(trained_model_path_FC)
        model_Fuse.load_state_dict(checkpoint['model_state_dict'])
        optimizer_fusion.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
    else:
        print('\n---------------------------Training Recognition FC Layer -----------------------------')
        validation_count = 0
        previous_loss = 0
        count = 0

        model_RNN.eval()
        model_CNN.eval()
        while True:
            count += 1
            model_Fuse.train()
            # training
            for batch_idx, data in enumerate(train_dataloader, 0):
                # zero gradients
                optimizer_fusion.zero_grad()

                # get training data
                inputs, labels = Variable(data["data"]).float(), Variable(data["label"]).long()
                inputs = inputs.to(device)
                labels = labels.squeeze()
                labels = labels.to(device)

                # get embedded features from RNN and CNN networks
                with torch.no_grad():
                    _, embedded_RNN = model_RNN(inputs)
                    inputs = inputs.unsqueeze(1)
                    _, embedded_CNN = model_CNN(inputs)
                embedded_RNN_CNN = torch.cat([embedded_RNN, embedded_CNN], dim=1)

                outputs, _ = model_Fuse(embedded_RNN_CNN)
                outputs = outputs.to(device)

                # loss calculation
                loss_RNN_CNN = F.cross_entropy(outputs, labels)
                if batch_idx % 500 == 0:
                    print(' {}/{} Train CNN-RNN-Fusion Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iTimes,
                        experiment_times,
                        count,
                        batch_idx * len(data),
                        len(train_dataloader.dataset),
                        100. * batch_idx / len(train_dataloader),
                        loss_RNN_CNN.data))

                loss_RNN_CNN.backward()
                optimizer_fusion.step()

            # validating
            curr_validate_loss = 0
            model_Fuse.eval()
            for batch_idx, data in enumerate(validation_dataloader, 0):
                # load test data
                data, target = Variable(data["data"]).float(), Variable(data["label"]).long()
                target = target.squeeze()
                target = target.to(device)
                data = data.to(device)

                # data = data.unsqueeze(1)
                # run model
                with torch.no_grad():
                    _, embedded_RNN = model_RNN(data)
                    data = data.unsqueeze(1)
                    _, embedded_CNN = model_CNN(data)
                    embedded_RNN_CNN = torch.cat([embedded_RNN, embedded_CNN], dim=1)

                    outputs, _ = model_Fuse(embedded_RNN_CNN)
                    outputs = outputs.to(device)

                    if target.dim() != 0 and output.dim() != 0:
                        # get loss and determine predict value
                        curr_validate_loss += F.cross_entropy(outputs, target, reduction='sum').data

            curr_validate_loss /= len(validation_dataloader.dataset)

            # stop if loss does not reduce for several times
            if previous_loss <= curr_validate_loss and previous_loss != 0:
                validation_count += 1
            else:
                validation_count = 0
                previous_loss = curr_validate_loss
                model_Fuse_store = model_Fuse

            print('--- Validation Loss: {:.7f} \t Count: {}'.format(
                curr_validate_loss.data,
                validation_count))

            if validation_count == validation_threshold_Fuse:
                model_Fuse = model_Fuse_store
                break
        torch.save({'epoch': count,
                    'model_state_dict': model_Fuse.state_dict(),
                    'optimizer_state_dict': optimizer_fusion.state_dict(),
                    'loss': previous_loss},
                   trained_model_path_FC + str(iTimes))

    ############################################################################
    # TESTING
    correct_CNN = 0
    correct_RNN = 0
    correct_RNN_CNN = 0

    all_recognition_case = 0

    all_score_output_RNN_CNN = torch.empty([])
    all_score_output_RNN = torch.empty([])
    all_score_output_CNN = torch.empty([])
    all_target = torch.empty([])

    # load test data
    testing_dataset = GaitSessionDataset(file_path=testing_file, data_length=data_length, dimension=dimension)
    testing_dataloader = DataLoader(testing_dataset, batch_size=32, shuffle=False)
    model_CNN.eval()
    model_RNN.eval()
    for batch_idx, data in enumerate(testing_dataloader, 0):
        data, target = Variable(data["data"]).float(), Variable(data["label"]).long()
        data = data.to(device)
        target = target.squeeze()
        target = target.to(device)

        # run model
        if target.dim() != 0:
            with torch.no_grad():
                if using_RNN:
                    output_RNN, _ = model_RNN(data)

                if using_CNN:
                    data = data.unsqueeze(1)
                    output_CNN, _ = model_CNN(data)
            output_RNN_CNN = output_RNN + output_CNN

            if batch_idx == 0:
                all_score_output_RNN_CNN = output_RNN_CNN
                all_score_output_RNN = output_RNN
                all_score_output_CNN = output_CNN
                all_target = target
            else:
                all_score_output_RNN = torch.cat([all_score_output_RNN, output_RNN], dim=0)
                all_score_output_CNN = torch.cat([all_score_output_CNN, output_CNN], dim=0)
                all_score_output_RNN_CNN = torch.cat([all_score_output_RNN_CNN, output_RNN_CNN], dim=0)

                all_target = torch.cat([all_target, target], dim=0)

            all_recognition_case += target.shape[0]
            if using_RNN:
                pred_RNN = output_RNN.data.max(1, keepdim=True)[1]
                correct_RNN += pred_RNN.eq(target.data.view_as(pred_RNN)).cpu().sum()

            if using_CNN:
                pred_CNN = output_CNN.data.max(1, keepdim=True)[1]
                correct_CNN += pred_CNN.eq(target.data.view_as(pred_CNN)).cpu().sum()

            pred_RNN_CNN = output_RNN_CNN.data.max(1, keepdim=True)[1]
            correct_RNN_CNN += pred_RNN_CNN.eq(target.data.view_as(pred_CNN)).cpu().sum()

    print('    + Classification result CNN - NO FUSION: {}/{} ({:.3f}%)'.format(
        correct_CNN, len(testing_dataloader.dataset),
        float(100. * correct_CNN) / all_recognition_case))

    print('    + Classification result RNN - NO FUSION: {}/{} ({:.3f}%)'.format(
        correct_RNN, len(testing_dataloader.dataset),
        float(100. * correct_RNN) / all_recognition_case))

    # print('    + Classification result CNN RNN Score FUSION: {}/{} ({:.3f}%)'.format(
    #     correct_RNN_CNN, len(testing_dataloader.dataset),
    #     float(100. * correct_RNN_CNN) / all_recognition_case))
    # print(' ----------------------------------------------------------------')


    # using fusion (cnn and rnn seperatedly)

    total_case_sequence_fusion = 0
    correct_CNN_sequence_fusion = 0
    correct_RNN_sequence_fusion = 0
    correct_RNN_CNN_sequence_fusion = 0

    # all_target = all_target.cpu()
    all_target_np = all_target.cpu().numpy()
    for iUser in range(user_number):
        # get all the templates of the current number
        idx = np.where(all_target_np == iUser)
        idx = idx[0]
        cur_user_score_output_RNN = all_score_output_RNN[idx][:]
        cur_user_score_output_CNN = all_score_output_CNN[idx][:]
        cur_user_score_output_RNN_CNN = all_score_output_RNN_CNN[idx][:]
        # cur_user_target = all_target[idx]

        # fuse the results and determine the decision
        iStart = 0
        iEnd = iStart + number_of_template

        while iEnd <= cur_user_score_output_RNN.shape[0]+1:
            multiple_results_CNN = cur_user_score_output_CNN[iStart:iEnd, :]
            multiple_results_RNN = cur_user_score_output_RNN[iStart:iEnd, :]
            multiple_results_RNN_CNN = cur_user_score_output_RNN_CNN[iStart:iEnd, :]
            if using_CNN:
                mean_results_CNN = torch.mean(multiple_results_CNN, 0, keepdim=True)
                pred_CNN = mean_results_CNN.data.max(1, keepdim=False)[1]

                if pred_CNN[0] == iUser:
                    correct_CNN_sequence_fusion += 1

            if using_RNN:
                mean_results_RNN = torch.mean(multiple_results_RNN, 0, keepdim=True)
                pred_RNN = mean_results_RNN.data.max(1, keepdim=True)[1]

                if pred_RNN[0] == iUser:
                    correct_RNN_sequence_fusion += 1

            mean_results_RNN_CNN = torch.mean(multiple_results_RNN_CNN, 0, keepdim=True)
            pred_RNN_CNN = mean_results_RNN_CNN.data.max(1, keepdim=True)[1]

            if pred_RNN_CNN[0] == iUser:
                correct_RNN_CNN_sequence_fusion += 1

            total_case_sequence_fusion += 1

            iStart = iStart + number_of_template
            iEnd = iStart + number_of_template

    print('    + Classification result CNN - FUSION: {}/{} ({:.3f}%)'.format(
        correct_CNN_sequence_fusion, total_case_sequence_fusion,
        float(100. * correct_CNN_sequence_fusion) / total_case_sequence_fusion))

    print('    + Classification result RNN - FUSION: {}/{} ({:.3f}%)'.format(
        correct_RNN_sequence_fusion, total_case_sequence_fusion,
        float(100. * correct_RNN_sequence_fusion) / total_case_sequence_fusion))

    print('    + Classification result RNN CNN - FUSION: {}/{} ({:.3f}%)'.format(
        correct_RNN_CNN_sequence_fusion, total_case_sequence_fusion,
        float(100. * correct_RNN_CNN_sequence_fusion) / total_case_sequence_fusion))
    print(' ----------------------------------------------------------------')

    # FEATURE FUSION
    # TESTING
    correct_RNN_CNN_1temp = 0
    correct_RNN_CNN_ALLTemp = 0

    all_recognition_case = 0
    all_score_output_RNN_CNN = torch.empty([])
    all_target = torch.empty([])

    # load test data
    model_CNN.eval()
    model_RNN.eval()
    model_Fuse_store.eval()

    for batch_idx, data in enumerate(testing_dataloader, 0):
        data, target = Variable(data["data"]).float(), Variable(data["label"]).long()
        data = data.to(device)
        target = target.squeeze()
        target = target.to(device)

        # run model
        if target.dim() != 0:
            with torch.no_grad():
                _, embedded_RNN = model_RNN(data)
                data = data.unsqueeze(1)
                _, embedded_CNN = model_CNN(data)

                embedded_RNN_CNN = torch.cat([embedded_RNN, embedded_CNN], dim=1)

                output_RNN_CNN, _ = model_Fuse(embedded_RNN_CNN)

                pred_RNN_CNN_fuse = output_RNN_CNN.data.max(1, keepdim=True)[1]
                correct_RNN_CNN_1temp += pred_RNN_CNN_fuse.eq(target.data.view_as(pred_RNN_CNN_fuse)).cpu().sum()

    print('    + Classification result RNN CNN - Feature FUSION: {}/{} ({:.3f}%)'.format(
        correct_RNN_CNN_1temp, len(testing_dataloader.dataset),
        float(100. * correct_RNN_CNN_1temp) / len(testing_dataloader.dataset)))

