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
experiment_times = 10        # number of experiments
validation_threshold_RNN = 15  # after #validation_threshold times that the validation loss does not decrease, the training process stops
learning_rate_RNN = 0.15                                                                              #^#

###########  NETWORK PARAMETERS  #################
data_length = 80       # number of signals per each gait cycle
dimension = 6           # number of channnels

rnn_input_dim_list = torch.IntTensor([5])
rnn_hidden_dim = 40   # the number of scalars that is returned from LSTM node at each stage       #^#
rnn_layer_dim = 2      # the number of layers in the RNN network                                    #^#

score_fusion = 1
number_of_template = 5
fusion_all_case = 1

classificatiton_result = torch.zeros([rnn_input_dim_list.shape[0], experiment_times])

##################  DATASET  ######################

file_path = "..\\Dataset\\segments\\"
file_name = 'A_WHU_equal_118_trn1519_ovl9.7_'

user_number = 118
rnn_output_dim = user_number

training_file = file_path + file_name + "train"
testing_file = file_path + file_name + "test"
validating_file = file_path + file_name + "vali"


bLoadTrainedModel = True   # load a trained model
trained_model_path = "..\\Dataset\\trained_models\\_gait_recognition_RNN_whu_rnn3"

if bLoadTrainedModel == False:
    training_dataset = GaitSessionDataset(file_path=training_file, data_length=data_length, dimension=dimension)
    validation_dataset = GaitSessionDataset(file_path=validating_file, data_length=data_length, dimension=dimension)

    train_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

testing_dataset = GaitSessionDataset(file_path=testing_file, data_length=data_length, dimension=dimension)
testing_dataloader = DataLoader(testing_dataset, batch_size=32, shuffle=False)

# # # # # # # # # # # # # # # #  TRAINING  # # # # # # # # # # # # # # # # # # # # # # #
for iInput_Dim in range(0, rnn_input_dim_list.shape[0]):
    rnn_input_dim = rnn_input_dim_list[iInput_Dim]

    for iTimes in range(0, experiment_times):
        #    ######## TRAIN RNN NETWORK ########
        model_RNN = LSTM_6Chan_OU(rnn_input_dim, rnn_hidden_dim, rnn_layer_dim, rnn_output_dim,
                                       segment_length=data_length)
        model_RNN_store = LSTM_6Chan_OU(rnn_input_dim, rnn_hidden_dim, rnn_layer_dim, rnn_output_dim,
                                             segment_length=data_length)
        model_RNN = model_RNN.to(device)

        optimizer_RNN = torch.optim.SGD(model_RNN.parameters(), lr=learning_rate_RNN, momentum=0.9)
        criterion_rnn = nn.CrossEntropyLoss()
        if bLoadTrainedModel:
            checkpoint = torch.load(trained_model_path)
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
                    inputs, labels = Variable(data["data"]).float().requires_grad_(), Variable(data["label"]).long()

                    inputs = inputs.to(device)
                    labels = labels.squeeze()
                    labels = labels.to(device)

                    # forward pass
                    outputs_RNN, _ = model_RNN(inputs)
                    outputs_RNN = outputs_RNN.to(device)

                    # loss calculation
                    loss = F.nll_loss(outputs_RNN, labels)

                    if batch_idx % 500 == 0:
                        print('Input dim {} {}/{} Train RNN Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            rnn_input_dim,
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
                    data = data.to(device)
                    # run model
                    with torch.no_grad():
                        output, _ = model_RNN(data)
                        output = output.cpu()

                        if target.dim() != 0 and output.dim() != 0:
                            # get loss and determine predict value
                            curr_validate_loss += F.nll_loss(output, target).data*data.shape[0]

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
               trained_model_path + '_rnn' + str(iTimes))

        ############################################################################
        # TESTING
        correct_RNN = 0
        # load test data

        model_RNN.eval()
        for batch_idx, data in enumerate(testing_dataloader, 0):
            data, target = Variable(data["data"]).float(), Variable(data["label"]).long()
            data = data.to(device)
            target = target.squeeze()
            target = target.to(device)

            # run model
            if target.dim() != 0:
                with torch.no_grad():
                    output_RNN, _ = model_RNN(data)
                    pred_RNN = output_RNN.data.max(1, keepdim=True)[1]
                    correct_RNN += pred_RNN.eq(target.data.view_as(pred_RNN)).cpu().sum()

        print('    + Classification result RNN - NO FUSION: {}/{} ({:.3f}%)'.format(
            correct_RNN, len(testing_dataloader.dataset),
            float(100. * correct_RNN) / len(testing_dataloader.dataset)))

        classificatiton_result[iInput_Dim, iTimes] = float(100. * correct_RNN) / len(testing_dataloader.dataset)


        save_path = "..\\Dataset\\trained_models\\whu_analyze_input_dim.txt"
        save_result = classificatiton_result.numpy()

        np.savetxt(save_path, save_result, delimiter=',')
