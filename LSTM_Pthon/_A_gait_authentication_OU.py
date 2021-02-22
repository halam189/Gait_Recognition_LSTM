import torch
import torch.tensor
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from authen_netw import *
from sklearn import svm

import math
import torch.nn.functional as F
from GaitDataset import *
# import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


############  TRAINING SETTINGS  #################

result = np.array([])
experiment_times = 1        # number of experiments
validation_threshold_RNN = 15  # after #validation_threshold times that the validation loss does not decrease, the training process stops
validation_threshold_CNN = 15

learning_rate_CNN = 0.00015
learning_rate_RNN = 0.15

using_CNN = 1
using_RNN = 1

###########  NETWORK PARAMETERS  #################

data_length = 100       # number of signals per each gait cycle
dimension = 6           # number of channnels

rnn_input_dim = 10     # the number of scalars that is fed to an LSTM node at each stage           #^#
rnn_hidden_dim = 40   # the number of scalars that is returned from LSTM node at each stage       #^#
rnn_layer_dim = 2      # the number of layers in the RNN network                                    #^#

kernel_rc = 'rbf'
gamma_rc = 0.1
nu_rc = 0.5

kernel = 'rbf'
gamma = 0.1
nu = 0.5

kernel_r = 'rbf'
gamma_r = 0.1
nu_r = 0.5


##################  DATASET  ######################
embedded_dim = 128
user_number = 520
user_number_testing = 225
rnn_output_dim = user_number

file_name = "..\\Dataset\\segments\\A_auth_OU_520_2251_ovl9.7_" # _auth_OU_520_2251_ovl9.7_"

training_file = file_name + "train"
testing_file = file_name + "test"
validating_file = file_name + "vali"

bLoadTrainedModel_RNN = True   # load a trained LSTM model
trained_model_path_RNN = "..\\Dataset\\trained_models\\_A_gait_authentication_RNN_OU_520_224"

bLoadTrainedModel_CNN = True   # load a trained CNN model
trained_model_path_CNN = "..\\Dataset\\trained_models\\_A_gait_authentication_CNN_OU_520_224"


if (not bLoadTrainedModel_RNN) and (not bLoadTrainedModel_CNN):
    training_dataset = GaitSessionTriplet(file_path=training_file, data_length=data_length, dimension=dimension,
                                          train=True)
    validation_dataset = GaitSessionTriplet(file_path=validating_file, data_length=data_length, dimension=dimension,
                                            train=True)

    train_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=True)


# # # # # # # # # # # # # # # #  TRAINING  # # # # # # # # # # # # # # # # # # # # # # #
for iTimes in range(0, experiment_times):
    #    ######## TRAIN or LOAD  RNN NETWORK ########
    model_RNN = LSTM_6Chan_Authen(rnn_input_dim, rnn_hidden_dim, rnn_layer_dim, rnn_output_dim,
                                             segment_length=data_length, embedding_dim=embedded_dim)
    model_RNN_store = LSTM_6Chan_Authen(rnn_input_dim, rnn_hidden_dim, rnn_layer_dim, rnn_output_dim,
                                                   segment_length=data_length, embedding_dim=embedded_dim)
    # model_RNN = torch.jit.script(model_RNN)
    model_RNN = model_RNN.to(device)

    if using_RNN:
        optimizer_RNN = torch.optim.SGD(model_RNN.parameters(), lr=learning_rate_RNN, momentum=0.9)
        criterion_rnn = torch.jit.script(TripletLoss())

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
            previous_loss = -1
            count = 0

            while True:
                count += 1
                model_RNN.train()
                # training
                for batch_idx, (anchor_sgm, positive_sgm, negative_smg, anchor_label) in enumerate(train_dataloader, 0):
                    # zero gradients
                    optimizer_RNN.zero_grad()
                    anchor_sgm = Variable(anchor_sgm)
                    positive_sgm = Variable(positive_sgm)
                    negative_smg = Variable(negative_smg)
                    anchor_label = Variable(anchor_label).long()

                    anchor_sgm = anchor_sgm.to(device)
                    positive_sgm = positive_sgm.to(device)
                    negative_smg = negative_smg.to(device)

                    anchor_out = model_RNN(anchor_sgm)
                    positive_out = model_RNN(positive_sgm)
                    negative_out = model_RNN(negative_smg)

                    loss = criterion_rnn(anchor_out, positive_out, negative_out)

                    if batch_idx % 100 == 0:
                        print(' {}/{} Train RNN Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            iTimes,
                            experiment_times,
                            count,
                            batch_idx * anchor_sgm.shape[0],
                            len(train_dataloader.dataset),
                            100. * batch_idx / len(train_dataloader),
                            loss.data))

                    loss.backward(retain_graph=True)
                    optimizer_RNN.step()

                # validating
                curr_validate_loss = 0
                model_RNN.eval()

                for batch_idx, (anchor_sgm, positive_sgm, negative_smg, anchor_label) in enumerate(
                        validation_dataloader, 0):
                    # zero gradients
                    with torch.no_grad():
                        anchor_sgm = anchor_sgm.to(device)
                        positive_sgm = positive_sgm.to(device)
                        negative_smg = negative_smg.to(device)

                        anchor_out = model_RNN(anchor_sgm)
                        positive_out = model_RNN(positive_sgm)
                        negative_out = model_RNN(negative_smg)

                        curr_validate_loss += criterion_rnn(anchor_out, positive_out, negative_out)

                curr_validate_loss /= len(validation_dataloader.dataset)

                # stop if loss does not reduce for several times
                if previous_loss <= curr_validate_loss and previous_loss != -1:
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
            # save trained model
            torch.save({'epoch': count,
                        'model_state_dict': model_RNN.state_dict(),
                        'optimizer_state_dict': optimizer_RNN.state_dict(),
                        'loss': previous_loss},
                       trained_model_path_RNN + str(iTimes))

    #    ######## TRAIN CNN NETWORK ########
    model_CNN = CNN_6Chan_Authen_OU(usernumber=user_number, embedding_dim=embedded_dim)
    model_CNN_store = CNN_6Chan_Authen_OU(usernumber=user_number, embedding_dim=embedded_dim)
    model_CNN = model_CNN.to(device)
    criterion_cnn = torch.jit.script(TripletLoss())

    if using_CNN:
        optimizer = torch.optim.Adam(model_CNN.parameters(), lr=learning_rate_CNN)
        train_embedded = np.zeros(0)
        train_label = np.zeros(0)

        if bLoadTrainedModel_CNN:
            checkpoint = torch.load(trained_model_path_CNN)
            model_CNN.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            loss = checkpoint['loss']
        else:
            print('\n---------------------------Training CNN -----------------------------')
            validation_count = 0
            previous_loss = -1
            count = 0
            while True:
                count += 1
                model_CNN.train()
                # training
                for batch_idx, (anchor_sgm, positive_sgm, negative_smg, anchor_label) in enumerate(train_dataloader, 0):
                    # zero gradients
                    optimizer.zero_grad()

                    anchor_sgm = Variable(anchor_sgm)
                    positive_sgm = Variable(positive_sgm)
                    negative_smg = Variable(negative_smg)
                    anchor_label = Variable(anchor_label).long()

                    anchor_sgm = anchor_sgm.unsqueeze(1).to(device)
                    positive_sgm = positive_sgm.unsqueeze(1).to(device)
                    negative_smg = negative_smg.unsqueeze(1).to(device)

                    anchor_out = model_CNN(anchor_sgm)
                    positive_out = model_CNN(positive_sgm)
                    negative_out = model_CNN(negative_smg)

                    loss = criterion_cnn(anchor_out, positive_out, negative_out)

                    if batch_idx % 500 == 0:
                        print(' {}/{} Train CNN Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            iTimes,
                            experiment_times,
                            count,
                            batch_idx * anchor_sgm.shape[0],
                            len(train_dataloader.dataset),
                            100. * batch_idx / len(train_dataloader),
                            loss.data))

                    loss.backward()
                    optimizer.step()

                # validating
                curr_validate_loss = 0
                model_CNN.eval()
                for batch_idx, (anchor_sgm, positive_sgm, negative_smg, anchor_label) in enumerate(
                        validation_dataloader, 0):
                    anchor_sgm = anchor_sgm.unsqueeze(1).to(device)
                    positive_sgm = positive_sgm.unsqueeze(1).to(device)
                    negative_smg = negative_smg.unsqueeze(1).to(device)
                    with torch.no_grad():
                        anchor_out = model_CNN(anchor_sgm)
                        positive_out = model_CNN(positive_sgm)
                        negative_out = model_CNN(negative_smg)

                        curr_validate_loss += criterion_cnn(anchor_out, positive_out, negative_out)

                curr_validate_loss /= len(validation_dataloader.dataset)

                # stop if loss does not reduce for several times
                if previous_loss <= curr_validate_loss and previous_loss != -1:
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
            # save trained model
            torch.save({'epoch': count,
                        'model_state_dict': model_CNN.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': previous_loss},
                       trained_model_path_CNN + str(iTimes))


    # TRAINING WITH OCSVM
    # get all user ID

    # threshold for SVM ROC curve
    # svm_score_threshold = np.array([0.00000000001, 0.0000000001, 0.000000001, 0.00000001, 0.0000001, 0.0000005, 0.000001, 0.0000015, 0.000002, 0.000003, 0.000004, 0.000005, 0.00001, 0.0001, 0.0005,
    #                                0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.005, 0.006, 0.007, 0.008,
    #                               0.009, 0.01, 0.0012, 0.0014, 0.0016, 0.02, 0.025, 0.03, 0.035, 0.05])
    testing_dataset = GaitSessionDataset(file_path=testing_file, data_length=data_length, dimension=dimension)
    testing_dataloader = DataLoader(testing_dataset, batch_size=32, shuffle=True)

    svm_score_threshold_0 = np.arange(0, 0.00001, 0.000001)
    svm_score_threshold_1 = np.arange(0.00002, 0.0002, 0.00001)
    svm_score_threshold_2 = np.arange(0.0004, 0.06, 0.0002)
    svm_score_threshold_3 = np.array(
        [0.07, 0.08, 0.09, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38])
    svm_score_threshold_4 = np.arange(0.4, 0.9, 0.001)

    svm_score_threshold = np.concatenate([svm_score_threshold_3, svm_score_threshold_4], 0)

    # RESULT FOR THRESHOLD
    total_false_reject = np.zeros(svm_score_threshold.size)
    total_false_accept = np.zeros(svm_score_threshold.size)

    total_false_reject_RNN = np.zeros(svm_score_threshold.size)
    total_false_accept_RNN = np.zeros(svm_score_threshold.size)

    total_false_reject_CNN = np.zeros(svm_score_threshold.size)
    total_false_accept_CNN = np.zeros(svm_score_threshold.size)

    # MAJOR SCORE FUSION
    total_false_reject_fuse_m = np.zeros(svm_score_threshold.size)
    total_false_accept_fuse_m = np.zeros(svm_score_threshold.size)

    total_false_reject_fuse_RNN_m = np.zeros(svm_score_threshold.size)
    total_false_accept_fuse_RNN_m = np.zeros(svm_score_threshold.size)

    total_false_reject_fuse_CNN_m = np.zeros(svm_score_threshold.size)
    total_false_accept_fuse_CNN_m = np.zeros(svm_score_threshold.size)

    total_imposter_trying = 0
    total_genuine_trying = 0

    total_imposter_trying_fuse = 0
    total_genuine_trying_fuse = 0

    all_user_target = torch.Tensor().long()

    all_user_score = np.empty([])
    all_user_predict = torch.Tensor().long()
    all_user_score_CNN = np.empty([])
    all_user_predict_CNN = torch.Tensor().long()
    all_user_score_RNN = np.empty([])
    all_user_predict_RNN = torch.Tensor().long()

    # recognition performance
    correct_CNN = 0
    correct_RNN = 0
    all_recognition_case = 0


    # extract features using the trained networks
    model_CNN.eval()
    model_RNN.eval()

    # embedded templates and the corresponding labels
    embedded_templates_CNN = torch.tensor([])  # gait features extracted from CNN
    embedded_templates_RNN = torch.tensor([])  # gait features extracted from RNN
    data_label = torch.LongTensor([])

    for batch_idx, data in enumerate(testing_dataloader, 0):
        # buffer for extracted templates
        embedded_templates_cur_batch_CNN = torch.tensor([])
        embedded_templates_cur_batch_RNN = torch.tensor([])

        # load test data
        data, target = Variable(data["data"]).float(), Variable(data["label"]).long()
        data = data.to(device)
        if target.dim() != 0:
            with torch.no_grad():
                if using_RNN:
                    embedding_RNN = model_RNN(data)
                    if embedding_RNN.dim() != 0:
                        embedded_templates_cur_batch_RNN = embedding_RNN.cpu()
                        embedded_templates_RNN = torch.cat(
                            [embedded_templates_RNN, embedded_templates_cur_batch_RNN], dim=0)

                if using_CNN:
                    data = data.unsqueeze(1)
                    embedding_CNN = model_CNN(data)
                    if embedding_CNN.dim() != 0:
                        embedded_templates_cur_batch_CNN = embedding_CNN.cpu()
                        embedded_templates_CNN = torch.cat(
                            [embedded_templates_CNN, embedded_templates_cur_batch_CNN], dim=0)

                data_label = torch.cat([data_label, target], dim=0)

    testing_label = data_label.data.numpy()
    # train_embedded = embedded_templates.data.numpy()
    testing_embedded_CNN = embedded_templates_CNN.data.numpy()
    testing_embedded_RNN = embedded_templates_RNN.data.numpy()

    # built a model for each user
    # for each user, divide his/her data to training and testing
    # training is used for construct the OSVM model
    # testing is used for infererring the model to determine the FRR
    # all other users are used for determine the FAR

    for curr_user in range(1, user_number_testing+1):  # train_label_ID:
        # print(curr_user)

        # get data of current user only
        idx = np.where(testing_label == curr_user)
        idx = idx[0]
        # curr_data_train = train_embedded[idx][:]
        curr_user_data_CNN = testing_embedded_CNN[idx][:]
        curr_user_data_RNN = testing_embedded_RNN[idx][:]

        # divide to training and testing
        number_of_template = np.shape(idx)[0]
        train_number = int(number_of_template / 2)

        curr_user_data_CNN_train = curr_user_data_CNN[0:train_number][:]
        curr_user_data_RNN_train = curr_user_data_RNN[0:train_number][:]

        curr_user_data_CNN_test = curr_user_data_CNN[train_number:][:]
        curr_user_data_RNN_test = curr_user_data_RNN[train_number:][:]

        # other users data
        idx_other = np.where(testing_label != curr_user)

        idx_other = idx_other[0]

        testing_label_other = testing_label[idx_other][:]
        other_users_CNN = testing_embedded_CNN[idx_other][:]
        other_users_RNN = testing_embedded_RNN[idx_other][:]

        curr_user_data_train = np.concatenate([curr_user_data_CNN_train, curr_user_data_RNN_train], axis=1)

        # train with OneClassSVM
        clf = svm.OneClassSVM(kernel=kernel_rc, gamma=gamma_rc, nu=nu_rc)
        clf.fit(curr_user_data_train)

        clf_CNN = svm.OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
        clf_CNN.fit(curr_user_data_CNN_train)

        clf_RNN = svm.OneClassSVM(kernel=kernel_r, gamma=gamma_r, nu=nu_r)
        clf_RNN.fit(curr_user_data_RNN_train)

        one_user_target = torch.Tensor().long().to(device)

        curr_user_data_test = np.concatenate([curr_user_data_CNN_test, curr_user_data_RNN_test], axis=1)
        other_users_data = np.concatenate([other_users_CNN, other_users_RNN], axis=1)

        # classify the test segments
        pred_label_curr_user = clf.predict(curr_user_data_test)
        distance = clf.decision_function(curr_user_data_test)
        score_curr_user = clf.score_samples(curr_user_data_test)

        pred_label_curr_user_CNN = clf_CNN.predict(curr_user_data_CNN_test)
        score_curr_user_CNN = clf_CNN.score_samples(curr_user_data_CNN_test)

        pred_label_curr_user_RNN = clf_RNN.predict(curr_user_data_RNN_test)
        score_curr_user_RNN = clf_RNN.score_samples(curr_user_data_RNN_test)

        pred_label_other_user = clf.predict(other_users_data)
        score_other_user = clf.score_samples(other_users_data)

        pred_label_other_user_CNN = clf_CNN.predict(other_users_CNN)
        score_other_user_CNN = clf_CNN.score_samples(other_users_CNN)

        pred_label_other_user_RNN = clf_RNN.predict(other_users_RNN)
        score_other_user_RNN = clf_RNN.score_samples(other_users_RNN)

        #   calculate the classification accuracy with auto OSVM MODE
        # total case
        total_imposter_trying += np.shape(other_users_RNN)[0]
        total_genuine_trying += np.shape(curr_user_data_RNN_test)[0]

        # ******************************************************
        # compute error rate based on defined threshold and score

        # NORMALIZE all scored to range 0-1
        score_min = min(score_curr_user.min(), score_other_user.min())
        score_max = max(score_curr_user.max(), score_other_user.max())
        score_curr_user = (score_curr_user - score_min) / (score_max - score_min)
        score_other_user = (score_other_user - score_min) / (score_max - score_min)

        score_min_CNN = min(score_curr_user_CNN.min(), score_other_user_CNN.min())
        score_max_CNN = max(score_curr_user_CNN.max(), score_other_user_CNN.max())
        score_curr_user_CNN = (score_curr_user_CNN - score_min_CNN) / (score_max_CNN - score_min_CNN)
        score_other_user_CNN = (score_other_user_CNN - score_min_CNN) / (score_max_CNN - score_min_CNN)

        score_min_RNN = min(score_curr_user_RNN.min(), score_other_user_RNN.min())
        score_max_RNN = max(score_curr_user_RNN.max(), score_other_user_RNN.max())
        score_curr_user_RNN = (score_curr_user_RNN - score_min_RNN) / (score_max_RNN - score_min_RNN)
        score_other_user_RNN = (score_other_user_RNN - score_min_RNN) / (score_max_RNN - score_min_RNN)

        # FUSION MULTIPLE RESULTS
        ### RNN and CNN
        # score_curr_user
        score_curr_user_mean = np.mean(score_curr_user, axis=0)

        total_genuine_trying_fuse += 1

        score_all_other_user_mean = torch.Tensor([])
        for iOther in range(user_number_testing):
            # testing_label_other
            all_other_user_i_idx = np.where(testing_label_other == iOther)
            all_other_user_i_idx = all_other_user_i_idx[0]
            if all_other_user_i_idx.shape[0] != 0:
                total_imposter_trying_fuse += 1
                # get all score of this user
                cur_other_score = score_other_user[all_other_user_i_idx]
                cur_other_score_mean = np.mean(cur_other_score, axis=0)
                if score_all_other_user_mean.shape[0] == 0:
                    score_all_other_user_mean = torch.Tensor([cur_other_score_mean])
                else:
                    score_all_other_user_mean = torch.cat([score_all_other_user_mean, torch.Tensor([cur_other_score_mean])], axis=0)

        score_all_other_user_mean = score_all_other_user_mean.numpy()

        ### RNN
        # score_curr_user
        score_curr_user_mean_RNN = np.mean(score_curr_user_RNN, axis=0)

        score_all_other_user_mean_RNN = torch.Tensor([])
        for iOther in range(user_number_testing):
            # testing_label_other
            all_other_user_i_idx = np.where(testing_label_other == iOther)
            all_other_user_i_idx = all_other_user_i_idx[0]
            if all_other_user_i_idx.shape[0] != 0:
                # get all score of this user
                cur_other_score_RNN = score_other_user[all_other_user_i_idx]
                cur_other_score_mean_RNN = np.mean(cur_other_score_RNN, axis=0)
                if score_all_other_user_mean_RNN.shape[0] == 0:
                    score_all_other_user_mean_RNN = torch.Tensor([cur_other_score_mean_RNN])
                else:
                    score_all_other_user_mean_RNN = torch.cat(
                        [score_all_other_user_mean_RNN, torch.Tensor([cur_other_score_mean_RNN])], axis=0)
        score_all_other_user_mean_RNN = score_all_other_user_mean_RNN.numpy()

        ### CNN
        # score_curr_user
        score_curr_user_mean_CNN = np.mean(score_curr_user_CNN, axis=0)

        score_all_other_user_mean_CNN = torch.Tensor([])
        for iOther in range(user_number_testing):
            # testing_label_other
            all_other_user_i_idx = np.where(testing_label_other == iOther)
            all_other_user_i_idx = all_other_user_i_idx[0]
            if all_other_user_i_idx.shape[0] != 0:
                # get all score of this user
                cur_other_score_CNN = score_other_user[all_other_user_i_idx]
                cur_other_score_mean_CNN = np.mean(cur_other_score_CNN, axis=0)
                if score_all_other_user_mean_CNN.shape[0] == 0:
                    score_all_other_user_mean_CNN = torch.Tensor([cur_other_score_mean_CNN])
                else:
                    score_all_other_user_mean_CNN = torch.cat(
                        [score_all_other_user_mean_CNN, torch.Tensor([cur_other_score_mean_CNN])], axis=0)
        score_all_other_user_mean_CNN = score_all_other_user_mean_CNN.numpy()

        # determine the accuracy for each threshhold
        for iThreshold in range(svm_score_threshold.size):
            # RNN CNN case
            false_reject = np.where(score_curr_user < svm_score_threshold[iThreshold])
            total_false_reject[iThreshold] += np.shape(np.array(false_reject)[0][:])[0]

            false_accept = np.where(score_other_user >= svm_score_threshold[iThreshold])
            total_false_accept[iThreshold] += np.shape(np.array(false_accept)[0][:])[0]

            # CNN case
            false_reject_cnn = np.where(score_curr_user_CNN < svm_score_threshold[iThreshold])
            total_false_reject_CNN[iThreshold] += np.shape(np.array(false_reject_cnn)[0][:])[0]

            false_accept_cnn = np.where(score_other_user_CNN >= svm_score_threshold[iThreshold])
            total_false_accept_CNN[iThreshold] += np.shape(np.array(false_accept_cnn)[0][:])[0]

            # RNN case
            false_reject_rnn = np.where(score_curr_user_RNN < svm_score_threshold[iThreshold])
            total_false_reject_RNN[iThreshold] += np.shape(np.array(false_reject_rnn)[0][:])[0]

            false_accept_rnn = np.where(score_other_user_RNN >= svm_score_threshold[iThreshold])
            total_false_accept_RNN[iThreshold] += np.shape(np.array(false_accept_rnn)[0][:])[0]

            # MAJOUR  SCORE FUSION
            # RNN CNN case
            false_reject = np.where(score_curr_user < svm_score_threshold[iThreshold])
            if np.shape(np.array(false_reject)[0][:])[0] > score_curr_user.size/2:
                total_false_reject_fuse_m[iThreshold] += 1

            for iOther in range(user_number_testing):
                # testing_label_other
                all_other_user_i_idx = np.where(testing_label_other == iOther)
                all_other_user_i_idx = all_other_user_i_idx[0]

                if all_other_user_i_idx.shape[0] != 0:
                    # get all score of this user
                    cur_other_score = score_other_user[all_other_user_i_idx]
                    false_accept_one_other = np.where(cur_other_score >= svm_score_threshold[iThreshold])
                    if np.shape(np.array(false_accept_one_other)[0][:])[0] > cur_other_score.size / 2:
                        total_false_accept_fuse_m[iThreshold] += 1

            # RNN case
            false_reject_rnn = np.where(score_curr_user_RNN < svm_score_threshold[iThreshold])
            if np.shape(np.array(false_reject_rnn)[0][:])[0] > score_curr_user_RNN.size / 2:
                total_false_reject_fuse_RNN_m[iThreshold] += 1

            for iOther in range(user_number_testing):
                # testing_label_other
                all_other_user_i_idx = np.where(testing_label_other == iOther)
                all_other_user_i_idx = all_other_user_i_idx[0]

                if all_other_user_i_idx.shape[0] != 0:
                    # get all score of this user
                    cur_other_score_RNN = score_other_user_RNN[all_other_user_i_idx]
                    false_accept_one_other_RNN = np.where(cur_other_score_RNN >= svm_score_threshold[iThreshold])
                    if np.shape(np.array(false_accept_one_other_RNN)[0][:])[0] > cur_other_score_RNN.size / 2:
                        total_false_accept_fuse_RNN_m[iThreshold] += 1

            # CNN case
            false_reject_CNN = np.where(score_curr_user_CNN < svm_score_threshold[iThreshold])
            if np.shape(np.array(false_reject_CNN)[0][:])[0] > score_curr_user_CNN.size / 2:
                total_false_reject_fuse_CNN_m[iThreshold] += 1

            for iOther in range(user_number_testing):
                # testing_label_other
                all_other_user_i_idx = np.where(testing_label_other == iOther)
                all_other_user_i_idx = all_other_user_i_idx[0]

                if all_other_user_i_idx.shape[0] != 0:
                    # get all score of this user
                    cur_other_score_CNN = score_other_user_CNN[all_other_user_i_idx]
                    false_accept_one_other_CNN = np.where(cur_other_score_CNN >= svm_score_threshold[iThreshold])
                    if np.shape(np.array(false_accept_one_other_CNN)[0][:])[0] > cur_other_score_CNN.size / 2:
                        total_false_accept_fuse_CNN_m[iThreshold] += 1

    print('Authentication result with different scoring threshold: RNN and CNN\n')
    EER = 100
    position = 0
    result = np.empty([svm_score_threshold.size + 1, 2], dtype=float)

    for iProba in range(svm_score_threshold.size):
        result[iProba + 1][0] = total_false_accept[iProba] * 100. / total_imposter_trying
        result[iProba + 1][1] = total_false_reject[iProba] * 100. / total_genuine_trying

        # print('{:.7f}:\t FAR: {:.4f} \t FRR: {:.4f}\n'.format(
        #    svm_score_threshold[iProba],
        #    total_false_accept[iProba] * 100. / total_imposter_trying,
        #    total_false_reject[iProba] * 100. / total_genuine_trying
        # ))
        cur_eer = abs(total_false_accept[iProba] * 100. / total_imposter_trying - total_false_reject[
            iProba] * 100. / total_genuine_trying)
        if cur_eer < EER:
            position = iProba
            EER = cur_eer

    equal_FAR = total_false_accept[position] * 100. / total_imposter_trying
    equal_FRR = total_false_reject[position] * 100. / total_genuine_trying

    print('RNN CNN Equal error rate at: {}, {:.7f}:\t FAR: {:.4f} \t FRR: {:.4f}\n'.format(
        position,
        svm_score_threshold[position],
        equal_FAR,
        equal_FRR
    ))
    result[0][0] = equal_FAR
    result[0][1] = equal_FRR

    save_path = "./authen_ou_" + str(user_number) + "_" + str(user_number_testing) + "_" + str(
        embedded_dim) + "_" + str(iTimes) + "rcnn.txt"

    np.savetxt(save_path, result, delimiter=',')

    ###################################################################################################################

    print('Authentication result with different scoring threshold: RNN \n')
    EER = 100
    position = 0
    result_RNN = np.empty([svm_score_threshold.size + 1, 2], dtype=float)

    for iProba in range(svm_score_threshold.size):
        result_RNN[iProba + 1][0] = total_false_accept_RNN[iProba] * 100. / total_imposter_trying
        result_RNN[iProba + 1][1] = total_false_reject_RNN[iProba] * 100. / total_genuine_trying

        cur_eer = abs(total_false_accept_RNN[iProba] * 100. / total_imposter_trying - total_false_reject_RNN[
            iProba] * 100. / total_genuine_trying)
        if cur_eer < EER:
            position = iProba
            EER = cur_eer

    equal_FAR = total_false_accept_RNN[position] * 100. / total_imposter_trying
    equal_FRR = total_false_reject_RNN[position] * 100. / total_genuine_trying

    print('RNN Equal error rate at: {}, {:.7f}:\t FAR: {:.4f} \t FRR: {:.4f}\n'.format(
        position,
        svm_score_threshold[position],
        equal_FAR,
        equal_FRR
    ))

    result_RNN[0][0] = equal_FAR
    result_RNN[0][1] = equal_FRR
    save_path = "./authen_ou_" + str(user_number) + "_" + str(user_number_testing) + "_" + str(
        embedded_dim) + "_" + str(iTimes) + "rnn.txt"

    np.savetxt(save_path, result_RNN, delimiter=',')
    ########################################################################################################
    print('Authentication result with different scoring threshold: CNN\n')
    EER = 100
    position = 0
    result_CNN = np.empty([svm_score_threshold.size + 1, 2], dtype=float)

    for iProba in range(svm_score_threshold.size):
        result_CNN[iProba + 1][0] = total_false_accept_CNN[iProba] * 100. / total_imposter_trying
        result_CNN[iProba + 1][1] = total_false_reject_CNN[iProba] * 100. / total_genuine_trying

        # print('{:.7f}:\t FAR: {:.4f} \t FRR: {:.4f}\n'.format(
        #    svm_score_threshold[iProba],
        #    total_false_accept_CNN[iProba] * 100. / total_imposter_trying,
        #    total_false_reject_CNN[iProba] * 100. / total_genuine_trying
        # ))
        cur_eer = abs(total_false_accept_CNN[iProba] * 100. / total_imposter_trying - total_false_reject_CNN[
            iProba] * 100. / total_genuine_trying)
        if cur_eer < EER:
            position = iProba
            EER = cur_eer

    equal_FAR = total_false_accept_CNN[position] * 100. / total_imposter_trying
    equal_FRR = total_false_reject_CNN[position] * 100. / total_genuine_trying

    print('CNN Equal error rate at: {}, {:.7f}:\t FAR: {:.4f} \t FRR: {:.4f}\n'.format(
        position,
        svm_score_threshold[position],
        equal_FAR,
        equal_FRR
    ))
    result_CNN[0][0] = equal_FAR
    result_CNN[0][1] = equal_FRR
    save_path = "./authen_ou_" + str(user_number) + "_" + str(user_number_testing) + "_" + str(
        embedded_dim) + "_" + str(iTimes) + "cnn.txt"

    np.savetxt(save_path, result_CNN, delimiter=',')


    #####################################################################################################3
    #
    #
    #
    # FUSION CASE MAJOUR VOTING
    #
    #
    #
    ######################################################################################################
    print('----------------------FUSION---------------')
    print('FUSION MAJOUR VOTING: Authentication result with different scoring threshold: RNN and CNN\n')
    EER = 100
    position = 0
    result = np.empty([svm_score_threshold.size + 1, 2], dtype=float)

    for iProba in range(svm_score_threshold.size):
        result[iProba + 1][0] = total_false_accept_fuse_m[iProba] * 100. / total_imposter_trying_fuse
        result[iProba + 1][1] = total_false_reject_fuse_m[iProba] * 100. / total_genuine_trying_fuse

        cur_eer = abs(total_false_accept_fuse_m[iProba] * 100. / total_imposter_trying_fuse - total_false_reject_fuse_m[
            iProba] * 100. / total_genuine_trying_fuse)
        if cur_eer < EER:
            position = iProba
            EER = cur_eer

    equal_FAR = total_false_accept_fuse_m[position] * 100. / total_imposter_trying_fuse
    equal_FRR = total_false_reject_fuse_m[position] * 100. / total_genuine_trying_fuse

    print('RNN CNN Equal error rate at: {}, {:.7f}:\t FAR: {:.4f} \t FRR: {:.4f}\n'.format(
        position,
        svm_score_threshold[position],
        equal_FAR,
        equal_FRR
    ))
    result[0][0] = equal_FAR
    result[0][1] = equal_FRR

    # save_path = "./authen_ou_" + str(user_number) + "_" + str(user_number_testing) + "_" + str(
    #     embedded_dim) + "_" + str(iTimes) + "rcnn_fuse_m.txt"
    #
    # np.savetxt(save_path, result, delimiter=',')

    ###################################################################################################################

    print('FUSE: Authentication result with different scoring threshold: RNN \n')
    EER = 100
    position = 0
    result_RNN = np.empty([svm_score_threshold.size + 1, 2], dtype=float)

    for iProba in range(svm_score_threshold.size):
        result_RNN[iProba + 1][0] = total_false_accept_fuse_RNN_m[iProba] * 100. / total_imposter_trying
        result_RNN[iProba + 1][1] = total_false_reject_fuse_RNN_m[iProba] * 100. / total_genuine_trying

        cur_eer = abs(
            total_false_accept_fuse_RNN_m[iProba] * 100. / total_imposter_trying_fuse - total_false_reject_fuse_RNN_m[
                iProba] * 100. / total_genuine_trying_fuse)
        if cur_eer < EER:
            position = iProba
            EER = cur_eer

    equal_FAR = total_false_accept_fuse_RNN_m[position] * 100. / total_imposter_trying_fuse
    equal_FRR = total_false_reject_fuse_RNN_m[position] * 100. / total_genuine_trying_fuse

    print('RNN Equal error rate at: {}, {:.7f}:\t FAR: {:.4f} \t FRR: {:.4f}\n'.format(
        position,
        svm_score_threshold[position],
        equal_FAR,
        equal_FRR
    ))

    result_RNN[0][0] = equal_FAR
    result_RNN[0][1] = equal_FRR
    # save_path = "./authen_ou_" + str(user_number) + "_" + str(user_number_testing) + "_" + str(
    #     embedded_dim) + "_" + str(iTimes) + "rnn_fuse_m.txt"
    #
    # np.savetxt(save_path, result_RNN, delimiter=',')
    ########################################################################################################
    print('Authentication result with different scoring threshold: CNN\n')
    EER = 100
    position = 0
    result_CNN = np.empty([svm_score_threshold.size + 1, 2], dtype=float)

    for iProba in range(svm_score_threshold.size):
        result_CNN[iProba + 1][0] = total_false_accept_fuse_CNN_m[iProba] * 100. / total_imposter_trying_fuse
        result_CNN[iProba + 1][1] = total_false_reject_fuse_CNN_m[iProba] * 100. / total_genuine_trying_fuse

        cur_eer = abs(
            total_false_accept_fuse_CNN_m[iProba] * 100. / total_imposter_trying_fuse - total_false_reject_fuse_CNN_m[
                iProba] * 100. / total_genuine_trying_fuse)
        if cur_eer < EER:
            position = iProba
            EER = cur_eer

    equal_FAR = total_false_accept_fuse_CNN_m[position] * 100. / total_imposter_trying_fuse
    equal_FRR = total_false_reject_fuse_CNN_m[position] * 100. / total_genuine_trying_fuse

    print('CNN Equal error rate at: {}, {:.7f}:\t FAR: {:.4f} \t FRR: {:.4f}\n'.format(
        position,
        svm_score_threshold[position],
        equal_FAR,
        equal_FRR
    ))
    result_CNN[0][0] = equal_FAR
    result_CNN[0][1] = equal_FRR
    # save_path = "./authen_ou_" + str(user_number) + "_" + str(user_number_testing) + "_" + str(
    #     embedded_dim) + "_" + str(iTimes) + "cnn_fuse_m.txt"
    #
    # np.savetxt(save_path, result_CNN, delimiter=',')
