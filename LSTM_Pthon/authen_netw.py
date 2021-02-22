import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_6Chan_OU(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, segment_length=100):
        super(LSTM_6Chan_OU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim

        rate = 0.5
        self.rnn_acc_1 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)
        self.rnn_acc_2 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)
        self.rnn_acc_3 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)

        self.rnn_gyr_1 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)
        self.rnn_gyr_2 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)
        self.rnn_gyr_3 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(int(hidden_dim * 6), output_dim)
        )

        self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # x: [batch_size, number of signal, number of channel/axis]
        x = x.view(x.shape[0], 6, -1)

        # acceleration channel 1
        acc_1 = x[:, 0, :]
        acc_1 = acc_1.unsqueeze(2)
        acc_1 = torch.reshape(acc_1, [acc_1.shape[0], int(acc_1.shape[1] / self.input_dim), self.input_dim])
        embedded_acc_1,_ = self.rnn_acc_1(acc_1)
        embedded_acc_1 = embedded_acc_1[:, acc_1.shape[1] - 1, :]

        # acceleration channel 2
        acc_2 = x[:, 1, :]
        acc_2 = acc_2.unsqueeze(2)
        acc_2 = torch.reshape(acc_2, [acc_2.shape[0], int(acc_2.shape[1] / self.input_dim), self.input_dim])
        embedded_acc_2, _= self.rnn_acc_2(acc_2)
        embedded_acc_2 = embedded_acc_2[:, acc_1.shape[1] - 1, :]

        # acceleration channel 3
        acc_3 = x[:, 2, :]
        acc_3 = acc_3.unsqueeze(2)
        acc_3 = torch.reshape(acc_3, [acc_3.shape[0], int(acc_3.shape[1] / self.input_dim), self.input_dim])
        embedded_acc_3, _ = self.rnn_acc_3(acc_3)
        embedded_acc_3 = embedded_acc_3[:, acc_1.shape[1] - 1, :]

        # gyroscope channel 1
        gyr_1 = x[:, 3, :]
        gyr_1 = gyr_1.unsqueeze(2)
        gyr_1 = torch.reshape(gyr_1, [gyr_1.shape[0], int(gyr_1.shape[1] / self.input_dim), self.input_dim])
        embedded_gyr_1, _ = self.rnn_gyr_1(gyr_1)
        embedded_gyr_1 = embedded_gyr_1[:, acc_1.shape[1] - 1, :]

        # gyroscope channel 2
        gyr_2 = x[:, 4, :]
        gyr_2 = gyr_2.unsqueeze(2)
        gyr_2 = torch.reshape(gyr_2, [gyr_2.shape[0], int(gyr_2.shape[1] / self.input_dim), self.input_dim])

        embedded_gyr_2, _ = self.rnn_gyr_2(gyr_2)
        embedded_gyr_2 = embedded_gyr_2[:, acc_1.shape[1] - 1, :]

        # gyroscope channel 3
        gyr_3 = x[:, 5, :]
        gyr_3 = gyr_3.unsqueeze(2)
        gyr_3 = torch.reshape(gyr_3, [gyr_3.shape[0], int(gyr_3.shape[1] / self.input_dim), self.input_dim])
        embedded_gyr_3, _ = self.rnn_gyr_3(gyr_3)
        embedded_gyr_3 = embedded_gyr_3[:, acc_1.shape[1] - 1, :]

        template = torch.cat(
            [embedded_acc_1, embedded_acc_2, embedded_acc_3, embedded_gyr_1, embedded_gyr_2, embedded_gyr_3], dim=1)
        template = self.drop(template)
        out = self.fc(template)

        out = F.log_softmax(out, 1)
        return out, template


class CNN_6Chan_OU(torch.nn.Module):

    def __init__(self, usernumber):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(CNN_6Chan_OU, self).__init__()

        self.user_number = usernumber

        self.conv1_acc = nn.Sequential(
            nn.Conv2d(1, 240, kernel_size=(1, 9), stride=(1, 1), padding=(0, 0))
            , nn.ReLU()
            , nn.BatchNorm2d(240)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv1_gyr = nn.Sequential(
            nn.Conv2d(1, 240, kernel_size=(1, 9), stride=(1, 1), padding=(0, 0))
            , nn.ReLU()
            , nn.BatchNorm2d(240)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv2 = nn.Sequential(
            nn.Conv2d(240, 300, kernel_size=(1, 7), stride=(1, 1), padding=(0, 1))
            , nn.ReLU()
            , nn.BatchNorm2d(300)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv3 = nn.Sequential(
            nn.Conv2d(300, 360, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0))
            , nn.ReLU()
            , nn.BatchNorm2d(360)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv4 = nn.Sequential(
            nn.Conv2d(360, 420, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
            , nn.ReLU()
            , nn.BatchNorm2d(420)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.avgpool = nn.Sequential(
            nn.AvgPool2d((1, 5), stride=(1, 5))
            , nn.Dropout2d(0.5)
        )

        self.fc1 = nn.Sequential(
             nn.Linear(2520, int(self.user_number *3/2))
             , nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(int(self.user_number*3/2) , self.user_number)
            )

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        # input x : 23 x 59049 x 1
        # expected conv1d input : minibatch_size x num_channel x width
        batch_size = x.size(0)
        x = x.view(x.shape[0], 1, 6, -1)

        gyr_data = x[:, :, 0:3, :]
        acc_data = x[:, :, 3:6, :]
        # x : batch_size x 1 x 900

        out_gyr = self.conv1_gyr(gyr_data)
        out_acc = self.conv1_acc(acc_data)
        out = torch.cat([out_gyr, out_acc], dim=3)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.avgpool(out)

        embedded = out.view(batch_size, -1)  # flatten the tensor
        out = self.fc1(embedded)
        out = self.fc2(out)
        out = F.log_softmax(out, 1)
        return out, embedded


class CNN_6Chan_WHU(torch.nn.Module):

    def __init__(self, usernumber):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(CNN_6Chan_WHU, self).__init__()

        self.user_number = usernumber

        self.conv1_acc = nn.Sequential(
            nn.Conv2d(1, 240, kernel_size=(1, 9), stride=(1, 1), padding=(0, 0))
            , nn.ReLU()
            , nn.BatchNorm2d(240)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv1_gyr = nn.Sequential(
            nn.Conv2d(1, 240, kernel_size=(1, 9), stride=(1, 1), padding=(0, 0))
            , nn.ReLU()
            , nn.BatchNorm2d(240)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv2 = nn.Sequential(
            nn.Conv2d(240, 300, kernel_size=(1, 7), stride=(1, 1), padding=(0, 1))
            , nn.ReLU()
            , nn.BatchNorm2d(300)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv3 = nn.Sequential(
            nn.Conv2d(300, 360, kernel_size=(1, 5), stride=(1, 1), padding=(0, 1))
            , nn.ReLU()
            , nn.BatchNorm2d(360)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv4 = nn.Sequential(
            nn.Conv2d(360, 420, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
            , nn.ReLU()
            , nn.BatchNorm2d(420)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.avgpool = nn.Sequential(
            nn.AvgPool2d((1, 4), stride=(1, 4))
            , nn.Dropout2d(0.5)
        )

        self.fc1 = nn.Sequential(
             nn.Linear(2520, int(self.user_number *3/2))
             , nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(int(self.user_number*3/2) , self.user_number)
            )

    def forward(self, x):
        # expected conv1d input : minibatch_size x num_channel x width
        batch_size = x.size(0)
        x = x.view(x.shape[0], 1, 6, -1)

        gyr_data = x[:, :, 0:3, :]
        acc_data = x[:, :, 3:6, :]
        # x : batch_size x 1 x 900

        out_gyr = self.conv1_gyr(gyr_data)
        out_acc = self.conv1_acc(acc_data)
        out = torch.cat([out_gyr, out_acc], dim=3)
        out = self.conv2(out)

        out = self.conv3(out)
        out = self.conv4(out)
        # embedded = out.view(batch_size, -1)
        out = self.avgpool(out)

        out = out.view(batch_size, -1)  # flatten the tensor
        embedded = out
        out = self.fc1(out)

        out = self.fc2(out)
        out = F.log_softmax(out, 1)
        return out, embedded


class RNN_CNN_Feature_Fusion(nn.Module):
    def __init__(self, input_dim, user_number):
        super(RNN_CNN_Feature_Fusion, self).__init__()
        self.input_dim = input_dim
        self.user_number = user_number

        self.fc = nn.Sequential(
            nn.Dropout2d(0.3)
            , nn.Linear(input_dim, user_number)
        )

    def forward(self, x):
        embedded_features = x
        out = self.fc(embedded_features)

        return out, embedded_features


class LSTM_6Chan_Authen(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, embedding_dim=128, segment_length=100):
        super(LSTM_6Chan_Authen, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim

        rate = 0.3
        self.rnn_acc_1 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate,  batch_first=True)
        self.rnn_acc_2 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)
        self.rnn_acc_3 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)

        self.rnn_gyr_1 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)
        self.rnn_gyr_2 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)
        self.rnn_gyr_3 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)

        self.fc = nn.Linear(int(hidden_dim*6), self.embedding_dim)

        self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # x: [batch_size, number of signal, number of channel/axis]
        x = x.view(x.shape[0], 6, -1)

        # acceleration channel 1
        acc_1 = x[:, 0, :]
        acc_1 = acc_1.unsqueeze(2)
        acc_1 = torch.reshape(acc_1, [acc_1.shape[0], int(acc_1.shape[1]/self.input_dim), self.input_dim])
        h_acc_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_acc_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_acc_1, (h_acc_n1, c_acc_n1) = self.rnn_acc_1(acc_1, (h_acc_1.detach(), c_acc_1.detach()))
        embedded_acc_1 = embedded_acc_1[:, acc_1.shape[1]-1, :]

        # acceleration channel 2
        acc_2 = x[:, 1, :]
        acc_2 = acc_2.unsqueeze(2)
        acc_2 = torch.reshape(acc_2, [acc_2.shape[0], int(acc_2.shape[1] / self.input_dim), self.input_dim])
        h_acc_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_acc_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_acc_2, (h_acc_n2, c_acc_n2) = self.rnn_acc_2(acc_2, (h_acc_2.detach(), c_acc_2.detach()))
        embedded_acc_2 = embedded_acc_2[:, acc_1.shape[1] - 1, :]

        # acceleration channel 3
        acc_3 = x[:, 2, :]
        # acc_3 = self.init_fc_3(acc_3)
        acc_3 = acc_3.unsqueeze(2)
        acc_3 = torch.reshape(acc_3, [acc_3.shape[0], int(acc_3.shape[1] / self.input_dim), self.input_dim])
        h_acc_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_acc_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_acc_3, (h_acc_n3, c_acc_n3) = self.rnn_acc_3(acc_3, (h_acc_3.detach(), c_acc_3.detach()))
        embedded_acc_3 = embedded_acc_3[:, acc_1.shape[1] - 1, :]

        # gyroscope channel 1
        gyr_1 = x[:, 3, :]
        gyr_1 = gyr_1.unsqueeze(2)
        gyr_1 = torch.reshape(gyr_1, [gyr_1.shape[0], int(gyr_1.shape[1] / self.input_dim), self.input_dim])
        h_gyr_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_gyr_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_gyr_1, (h_gyr_n1, c_gyr_n1) = self.rnn_gyr_1(gyr_1, (h_gyr_1.detach(), c_gyr_1.detach()))
        embedded_gyr_1 = embedded_gyr_1[:, acc_1.shape[1] - 1, :]

        # gyroscope channel 2
        gyr_2 = x[:, 4, :]
        gyr_2 = gyr_2.unsqueeze(2)
        gyr_2 = torch.reshape(gyr_2, [gyr_2.shape[0], int(gyr_2.shape[1] / self.input_dim), self.input_dim])
        h_gyr_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_gyr_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_gyr_2, (h_gyr_n2, c_gyr_n2) = self.rnn_gyr_2(gyr_2, (h_gyr_2.detach(), c_gyr_2.detach()))
        embedded_gyr_2 = embedded_gyr_2[:, acc_1.shape[1] - 1, :]

        # gyroscope channel 3
        gyr_3 = x[:, 5, :]
        gyr_3 = gyr_3.unsqueeze(2)
        gyr_3 = torch.reshape(gyr_3, [gyr_3.shape[0], int(gyr_3.shape[1] / self.input_dim), self.input_dim])
        h_gyr_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_gyr_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_gyr_3, (h_gyr_n3, c_gyr_n3) = self.rnn_gyr_3(gyr_3, (h_gyr_3.detach(), c_gyr_3.detach()))
        embedded_gyr_3 = embedded_gyr_3[:, acc_1.shape[1] - 1, :]

        template = torch.cat([embedded_acc_1, embedded_acc_2, embedded_acc_3, embedded_gyr_1, embedded_gyr_2, embedded_gyr_3], dim=1)
        template = self.drop(template)
        out = self.fc(template)

        return out


class CNN_6Chan_Authen_OU(torch.nn.Module):

    def __init__(self, usernumber, embedding_dim=100):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(CNN_6Chan_Authen_OU, self).__init__()

        self.user_number = usernumber
        self.embedding_dim=embedding_dim

        self.conv1_acc = nn.Sequential(
            nn.Conv2d(1, 240, kernel_size=(1, 9), stride=(1, 1), padding=(0, 0))
            , nn.ReLU()
            , nn.BatchNorm2d(240)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv1_gyr = nn.Sequential(
            nn.Conv2d(1, 240, kernel_size=(1, 9), stride=(1, 1), padding=(0, 0))
            , nn.ReLU()
            , nn.BatchNorm2d(240)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv2 = nn.Sequential(
            nn.Conv2d(240, 300, kernel_size=(1, 7), stride=(1, 1), padding=(0, 1))
            , nn.ReLU()
            , nn.BatchNorm2d(300)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv3 = nn.Sequential(
            nn.Conv2d(300, 360, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0))
            , nn.ReLU()
            , nn.BatchNorm2d(360)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv4 = nn.Sequential(
            nn.Conv2d(360, 420, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
            , nn.ReLU()
            , nn.BatchNorm2d(420)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.avgpool = nn.Sequential(
            nn.AvgPool2d((1, 5), stride=(1, 5))
            , nn.Dropout2d(0.5)
        )

        self.fc1 = nn.Sequential(
             nn.Linear(2520, int(self.user_number *3/2))
             , nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(int(self.user_number*3/2) , self.embedding_dim)
            )

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        # input x : 23 x 59049 x 1
        # expected conv1d input : minibatch_size x num_channel x width
        batch_size = x.size(0)
        x = x.view(x.shape[0], 1, 6, -1)

        gyr_data = x[:, :, 0:3, :]
        acc_data = x[:, :, 3:6, :]

        out_gyr = self.conv1_gyr(gyr_data)
        out_acc = self.conv1_acc(acc_data)
        out = torch.cat([out_gyr, out_acc], dim=3)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.avgpool(out)

        embedded = out.view(batch_size, -1)  # flatten the tensor
        out = self.fc1(embedded)
        out = self.fc2(out)

        return out


class CNN_6Chan_Authen_WHU(torch.nn.Module):

    def __init__(self, usernumber, embedding_dim):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(CNN_6Chan_Authen_WHU, self).__init__()

        self.user_number = usernumber
        self.embedding_dim = embedding_dim

        self.conv1_acc = nn.Sequential(
            nn.Conv2d(1, 240, kernel_size=(1, 9), stride=(1, 1), padding=(0, 0))
            , nn.ReLU()
            , nn.BatchNorm2d(240)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv1_gyr = nn.Sequential(
            nn.Conv2d(1, 240, kernel_size=(1, 9), stride=(1, 1), padding=(0, 0))
            , nn.ReLU()
            , nn.BatchNorm2d(240)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv2 = nn.Sequential(
            nn.Conv2d(240, 300, kernel_size=(1, 7), stride=(1, 1), padding=(0, 1))
            , nn.ReLU()
            , nn.BatchNorm2d(300)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv3 = nn.Sequential(
            nn.Conv2d(300, 360, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0))
            , nn.ReLU()
            , nn.BatchNorm2d(360)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv4 = nn.Sequential(
            nn.Conv2d(360, 420, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
            , nn.ReLU()
            , nn.BatchNorm2d(420)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.avgpool = nn.Sequential(
            nn.AvgPool2d((1, 5), stride=(1, 5))
            , nn.Dropout2d(0.5)
        )

        self.fc1 = nn.Sequential(
             nn.Linear(1260, int(self.user_number * 3/2))
             , nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(int(self.user_number*3/2), self.embedding_dim)
            )

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        # input x : 23 x 59049 x 1
        # expected conv1d input : minibatch_size x num_channel x width
        batch_size = x.size(0)
        x = x.view(x.shape[0], 1, 6, -1)

        gyr_data = x[:, :, 0:3, :]
        acc_data = x[:, :, 3:6, :]
        # x : batch_size x 1 x 900

        out_gyr = self.conv1_gyr(gyr_data)
        out_acc = self.conv1_acc(acc_data)
        out = torch.cat([out_gyr, out_acc], dim=3)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.avgpool(out)

        embedded = out.view(batch_size, -1)  # flatten the tensor
        out = self.fc1(embedded)
        out = self.fc2(out)

        return out


#####################################################################################################################################


class RNN_CNN_Feature_Fusion_OUbk(nn.Module):
    def __init__(self, input_dim, user_number):
        super(RNN_CNN_Feature_Fusion_OUbk, self).__init__()
        self.input_dim = input_dim
        self.user_number = user_number

        self.fc1 = nn.Sequential(
            # nn.Dropout2d(0.5)
            nn.Linear(input_dim, int(user_number*3/2))
            # , nn.ReLU()
            , nn.Dropout2d(0.5)
        )

        self.fc2 = nn.Linear(int(user_number*3/2), user_number)

        self.fc = nn.Sequential(
            nn.Dropout2d(0.3)
            , nn.Linear(input_dim, user_number)
        )

    def forward(self, x):
        if (0):
            embedded_features = self.fc1(x)
            out = self.fc2(embedded_features)
        else:
            embedded_features = x
            out = self.fc(embedded_features)

        return out, embedded_features


class RNN_CNN_Feature_Fusion_WHU(nn.Module):
    def __init__(self, input_dim, user_number):
        super(RNN_CNN_Feature_Fusion_WHU, self).__init__()
        self.input_dim = input_dim
        self.user_number = user_number

        self.fc1 = nn.Sequential(
            # nn.Dropout2d(0.5)
            nn.Linear(input_dim, int(user_number*3/2))
            # , nn.ReLU()
            , nn.Dropout2d(0.5)
        )

        self.fc2 = nn.Linear(int(user_number*3/2), user_number)

        self.fc = nn.Sequential(
            nn.Dropout2d(0.3)
            , nn.Linear(input_dim, user_number)
        )

    def forward(self, x):
        if (0):
            embedded_features = self.fc1(x)
            out = self.fc2(embedded_features)
        else:
            embedded_features = x
            out = self.fc(embedded_features)

        return out, embedded_features


class RNN_CNN_Feature_Fusion_WHU_80(nn.Module):
    def __init__(self, input_dim, user_number):
        super(RNN_CNN_Feature_Fusion_WHU_80, self).__init__()
        self.input_dim = input_dim
        self.user_number = user_number

        self.fc1 = nn.Sequential(
            # nn.Dropout2d(0.5)
            nn.Linear(input_dim, int(user_number*3/2))
            # , nn.ReLU()
            , nn.Dropout2d(0.5)
        )

        self.fc2 = nn.Linear(int(user_number*3/2), user_number)

        self.fc = nn.Sequential(
            nn.Dropout2d(0.3)
            , nn.Linear(input_dim, user_number)
        )

    def forward(self, x):
        if (0):
            embedded_features = self.fc1(x)
            out = self.fc2(embedded_features)
        else:
            embedded_features = x
            out = self.fc(embedded_features)

        return out, embedded_features


class RNN_CNN_Feature_Fusion_OU_6_100(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, user_number, cycle_length=100):
        super(RNN_CNN_Feature_Fusion_OU_6_100, self).__init__()
        # RNN parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.user_number = user_number

        self.init_fc_1 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            # , nn.ReLU()
        )
        self.init_fc_2 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            # , nn.ReLU()
        )
        self.init_fc_3 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            # , nn.ReLU()
        )
        self.init_fc_4 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            # , nn.ReLU()
        )
        self.init_fc_5 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            # , nn.ReLU()
        )
        self.init_fc_6 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            # , nn.ReLU()
        )

        rnn_dropout_rate = 0.5
        self.rnn_acc_1 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rnn_dropout_rate,  batch_first=True)
        self.rnn_acc_2 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rnn_dropout_rate, batch_first=True)
        self.rnn_acc_3 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rnn_dropout_rate, batch_first=True)

        self.rnn_gyr_1 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rnn_dropout_rate, batch_first=True)
        self.rnn_gyr_2 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rnn_dropout_rate, batch_first=True)
        self.rnn_gyr_3 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rnn_dropout_rate, batch_first=True)

        # Readout layer
        # self.fc1 = nn.Linear(int(hidden_dim*6), int(output_dim))
        # self.fc2 = nn.Linear(int(output_dim), output_dim)
        # self.fc = nn.Linear(int(hidden_dim*6), output_dim)

        self.drop_CNN = nn.Dropout2d(0.5)

        ######################################################################################################
        # CNN layers
        self.conv1_acc = nn.Sequential(
            nn.Conv2d(1, 240, kernel_size=(1, 9), stride=(1, 1), padding=(0, 0))
            , nn.ReLU()
            , nn.BatchNorm2d(240)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv1_gyr = nn.Sequential(
            nn.Conv2d(1, 240, kernel_size=(1, 9), stride=(1, 1), padding=(0, 0))
            , nn.ReLU()
            , nn.BatchNorm2d(240)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv2 = nn.Sequential(
            nn.Conv2d(240, 300, kernel_size=(1, 7), stride=(1, 1), padding=(0, 1))
            , nn.ReLU()
            , nn.BatchNorm2d(300)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv3 = nn.Sequential(
            nn.Conv2d(300, 360, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0))
            , nn.ReLU()
            , nn.BatchNorm2d(360)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv4 = nn.Sequential(
            nn.Conv2d(360, 420, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
            , nn.ReLU()
            , nn.BatchNorm2d(420)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.avgpool = nn.Sequential(
            nn.AvgPool2d((1, 5), stride=(1, 5))
            , nn.Dropout2d(0.5)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(int(hidden_dim*6) + 2520, int(self.user_number * 3 / 2))
            , nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(int(self.user_number * 3 / 2), self.user_number)
        )

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        batch_size = x.size(0)
        x = x.view(x.shape[0], 1, 6, -1)

        # CNN
        gyr_data = x[:, :, 0:3, :]
        acc_data = x[:, :, 3:6, :]
        # x : batch_size x 1 x 900

        out_gyr = self.conv1_gyr(gyr_data)
        out_acc = self.conv1_acc(acc_data)
        out_CNN = torch.cat([out_gyr, out_acc], dim=3)
        out_CNN = self.conv2(out_CNN)
        out_CNN = self.conv3(out_CNN)
        out_CNN = self.conv4(out_CNN)
        out_CNN = self.avgpool(out_CNN)

        out_CNN = out_CNN.view(batch_size, -1)  # flatten the tensor



        # RNN process

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # x: [batch_size, number of signal, number of channel/axis]
        x = x.view(x.shape[0], 6, -1)

        # acceleration channel 1
        acc_1 = x[:, 0, :]
        # acc_1 = self.init_fc_1(acc_1)
        acc_1 = acc_1.unsqueeze(2)
        acc_1 = torch.reshape(acc_1, [acc_1.shape[0], int(acc_1.shape[1]/self.input_dim), self.input_dim])
        h_acc_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_acc_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_acc_1, (h_acc_n1, c_acc_n1) = self.rnn_acc_1(acc_1, (h_acc_1.detach(), c_acc_1.detach()))
        embedded_acc_1 = embedded_acc_1[:, acc_1.shape[1]-1, :]

        # acceleration channel 2
        acc_2 = x[:, 1, :]
        # acc_2 = self.init_fc_2(acc_2)
        acc_2 = acc_2.unsqueeze(2)
        acc_2 = torch.reshape(acc_2, [acc_2.shape[0], int(acc_2.shape[1] / self.input_dim), self.input_dim])
        h_acc_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_acc_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_acc_2, (h_acc_n2, c_acc_n2) = self.rnn_acc_2(acc_2, (h_acc_2.detach(), c_acc_2.detach()))
        embedded_acc_2 = embedded_acc_2[:, acc_1.shape[1] - 1, :]

        # acceleration channel 3
        acc_3 = x[:, 2, :]
        # acc_3 = self.init_fc_3(acc_3)
        acc_3 = acc_3.unsqueeze(2)
        acc_3 = torch.reshape(acc_3, [acc_3.shape[0], int(acc_3.shape[1] / self.input_dim), self.input_dim])
        h_acc_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_acc_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_acc_3, (h_acc_n3, c_acc_n3) = self.rnn_acc_3(acc_3, (h_acc_3.detach(), c_acc_3.detach()))
        embedded_acc_3 = embedded_acc_3[:, acc_1.shape[1] - 1, :]

        # gyroscope channel 1
        gyr_1 = x[:, 3, :]
        # gyr_1 = self.init_fc_4(gyr_1)
        gyr_1 = gyr_1.unsqueeze(2)
        gyr_1 = torch.reshape(gyr_1, [gyr_1.shape[0], int(gyr_1.shape[1] / self.input_dim), self.input_dim])
        h_gyr_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_gyr_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_gyr_1, (h_gyr_n1, c_gyr_n1) = self.rnn_gyr_1(gyr_1, (h_gyr_1.detach(), c_gyr_1.detach()))
        embedded_gyr_1 = embedded_gyr_1[:, acc_1.shape[1] - 1, :]

        # gyroscope channel 2
        gyr_2 = x[:, 4, :]
        # gyr_2 = self.init_fc_5(gyr_2)
        gyr_2 = gyr_2.unsqueeze(2)
        gyr_2 = torch.reshape(gyr_2, [gyr_2.shape[0], int(gyr_2.shape[1] / self.input_dim), self.input_dim])
        h_gyr_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_gyr_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_gyr_2, (h_gyr_n2, c_gyr_n2) = self.rnn_gyr_2(gyr_2, (h_gyr_2.detach(), c_gyr_2.detach()))
        embedded_gyr_2 = embedded_gyr_2[:, acc_1.shape[1] - 1, :]

        # gyroscope channel 3
        gyr_3 = x[:, 5, :]
        # gyr_3 = self.init_fc_6(gyr_3)
        gyr_3 = gyr_3.unsqueeze(2)
        gyr_3 = torch.reshape(gyr_3, [gyr_3.shape[0], int(gyr_3.shape[1] / self.input_dim), self.input_dim])
        h_gyr_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_gyr_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_gyr_3, (h_gyr_n3, c_gyr_n3) = self.rnn_gyr_3(gyr_3, (h_gyr_3.detach(), c_gyr_3.detach()))
        embedded_gyr_3 = embedded_gyr_3[:, acc_1.shape[1] - 1, :]

        out_RNN = torch.cat([embedded_acc_1, embedded_acc_2, embedded_acc_3, embedded_gyr_1, embedded_gyr_2, embedded_gyr_3], dim=1)
        # template = torch.cat([embedded_gyr_1, embedded_gyr_2, embedded_gyr_3], dim=1)

        # concatenate RNN and CNN output
        out = torch.cat([out_CNN, out_RNN], dim=1)

        # fully connected layer for recognition
        out = self.drop(out)
        embedded = self.fc1(out)
        out = self.fc2(embedded)

        out = F.log_softmax(out, 1)
        return out, embedded


class LSTMSeparatedMtO_6Chan_OU_TL1(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, embedding_dim=128, segment_length=100):
        super(LSTMSeparatedMtO_6Chan_OU_TL1, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim

        # self.init_fc_1 = nn.Sequential(
        #     nn.Linear(cycle_length, cycle_length)
        #     # , nn.ReLU()
        # )
        # self.init_fc_2 = nn.Sequential(
        #     nn.Linear(cycle_length, cycle_length)
        #     # , nn.ReLU()
        # )
        # self.init_fc_3 = nn.Sequential(
        #     nn.Linear(cycle_length, cycle_length)
        #     # , nn.ReLU()
        # )
        # self.init_fc_4 = nn.Sequential(
        #     nn.Linear(cycle_length, cycle_length)
        #     # , nn.ReLU()
        # )
        # self.init_fc_5 = nn.Sequential(
        #     nn.Linear(cycle_length, cycle_length)
        #     # , nn.ReLU()
        # )
        # self.init_fc_6 = nn.Sequential(
        #     nn.Linear(cycle_length, cycle_length)
        #     # , nn.ReLU()
        # )
        rate = 0.3
        self.rnn_acc_1 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate,  batch_first=True)
        self.rnn_acc_2 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)
        self.rnn_acc_3 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)

        self.rnn_gyr_1 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)
        self.rnn_gyr_2 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)
        self.rnn_gyr_3 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)

        # Readout layer
        # self.fc1 = nn.Sequential(
        #     nn.Linear(int(hidden_dim*6), int(output_dim*3/2))
        #     #,nn.ReLU()
        # )
        #
        # self.fc2 = nn.Linear(int(output_dim*3/2), output_dim)
        self.fc = nn.Linear(int(hidden_dim*6), self.embedding_dim)

        self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # x: [batch_size, number of signal, number of channel/axis]
        x = x.view(x.shape[0], 6, -1)

        # acceleration channel 1
        acc_1 = x[:, 0, :]
        # acc_1 = self.init_fc_1(acc_1)
        acc_1 = acc_1.unsqueeze(2)
        acc_1 = torch.reshape(acc_1, [acc_1.shape[0], int(acc_1.shape[1]/self.input_dim), self.input_dim])
        h_acc_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_acc_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_acc_1, (h_acc_n1, c_acc_n1) = self.rnn_acc_1(acc_1, (h_acc_1.detach(), c_acc_1.detach()))
        # embedded_acc_1, _ = self.rnn_acc_1(acc_1)  # , (h_acc_1.detach(), c_acc_1.detach()))
        # embedded_acc_1 = torch.reshape(embedded_acc_1, [batch_size, -1])
        embedded_acc_1 = embedded_acc_1[:, acc_1.shape[1]-1, :]

        # acceleration channel 2
        acc_2 = x[:, 1, :]
        # acc_2 = self.init_fc_2(acc_2)
        acc_2 = acc_2.unsqueeze(2)
        acc_2 = torch.reshape(acc_2, [acc_2.shape[0], int(acc_2.shape[1] / self.input_dim), self.input_dim])
        h_acc_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_acc_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_acc_2, (h_acc_n2, c_acc_n2) = self.rnn_acc_2(acc_2, (h_acc_2.detach(), c_acc_2.detach()))
        # embedded_acc_2, _ = self.rnn_acc_2(acc_2)
        # embedded_acc_2 = torch.reshape(embedded_acc_2, [batch_size, -1])
        embedded_acc_2 = embedded_acc_2[:, acc_1.shape[1] - 1, :]

        # acceleration channel 3
        acc_3 = x[:, 2, :]
        # acc_3 = self.init_fc_3(acc_3)
        acc_3 = acc_3.unsqueeze(2)
        acc_3 = torch.reshape(acc_3, [acc_3.shape[0], int(acc_3.shape[1] / self.input_dim), self.input_dim])
        h_acc_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_acc_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_acc_3, (h_acc_n3, c_acc_n3) = self.rnn_acc_3(acc_3, (h_acc_3.detach(), c_acc_3.detach()))
        # embedded_acc_3, _ = self.rnn_acc_3(acc_3)
        # embedded_acc_3 = torch.reshape(embedded_acc_3, [batch_size, -1])
        embedded_acc_3 = embedded_acc_3[:, acc_1.shape[1] - 1, :]

        # gyroscope channel 1
        gyr_1 = x[:, 3, :]
        # gyr_1 = self.init_fc_4(gyr_1)
        gyr_1 = gyr_1.unsqueeze(2)
        gyr_1 = torch.reshape(gyr_1, [gyr_1.shape[0], int(gyr_1.shape[1] / self.input_dim), self.input_dim])
        h_gyr_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_gyr_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_gyr_1, (h_gyr_n1, c_gyr_n1) = self.rnn_gyr_1(gyr_1, (h_gyr_1.detach(), c_gyr_1.detach()))
        # embedded_gyr_1, _ = self.rnn_gyr_1(gyr_1)
        # embedded_gyr_1 = torch.reshape(embedded_gyr_1, [batch_size, -1])
        embedded_gyr_1 = embedded_gyr_1[:, acc_1.shape[1] - 1, :]

        # gyroscope channel 2
        gyr_2 = x[:, 4, :]
        # gyr_2 = self.init_fc_5(gyr_2)
        gyr_2 = gyr_2.unsqueeze(2)
        gyr_2 = torch.reshape(gyr_2, [gyr_2.shape[0], int(gyr_2.shape[1] / self.input_dim), self.input_dim])
        h_gyr_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_gyr_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_gyr_2, (h_gyr_n2, c_gyr_n2) = self.rnn_gyr_2(gyr_2, (h_gyr_2.detach(), c_gyr_2.detach()))
        # embedded_gyr_2, _ = self.rnn_gyr_2(gyr_2)
        # embedded_gyr_2 = torch.reshape(embedded_gyr_2, [batch_size, -1])
        embedded_gyr_2 = embedded_gyr_2[:, acc_1.shape[1] - 1, :]

        # gyroscope channel 3
        gyr_3 = x[:, 5, :]
        # gyr_3 = self.init_fc_6(gyr_3)
        gyr_3 = gyr_3.unsqueeze(2)
        gyr_3 = torch.reshape(gyr_3, [gyr_3.shape[0], int(gyr_3.shape[1] / self.input_dim), self.input_dim])
        h_gyr_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_gyr_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_gyr_3, (h_gyr_n3, c_gyr_n3) = self.rnn_gyr_3(gyr_3, (h_gyr_3.detach(), c_gyr_3.detach()))
        # embedded_gyr_3, _ = self.rnn_gyr_3(gyr_3)
        # embedded_gyr_3 = torch.reshape(embedded_gyr_3, [batch_size, -1])
        embedded_gyr_3 = embedded_gyr_3[:, acc_1.shape[1] - 1, :]

        template = torch.cat([embedded_acc_1, embedded_acc_2, embedded_acc_3, embedded_gyr_1, embedded_gyr_2, embedded_gyr_3], dim=1)
        # template = torch.cat([embedded_gyr_1, embedded_gyr_2, embedded_gyr_3], dim=1)
        template = self.drop(template)
        # if (0):
        #     out = self.fc1(template)
        #     out = self.fc2(out)
        # else:
        out = self.fc(template)

        # out = F.log_softmax(out, 1)
        return out



class LSTMSeparatedMtO_6Chan_OU(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, cycle_length=100):
        super(LSTMSeparatedMtO_6Chan_OU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim

        self.init_fc_1 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            # , nn.ReLU()
        )
        self.init_fc_2 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            # , nn.ReLU()
        )
        self.init_fc_3 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            # , nn.ReLU()
        )
        self.init_fc_4 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            # , nn.ReLU()
        )
        self.init_fc_5 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            # , nn.ReLU()
        )
        self.init_fc_6 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            # , nn.ReLU()
        )
        rate = 0.3
        self.rnn_acc_1 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate,  batch_first=True)
        self.rnn_acc_2 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)
        self.rnn_acc_3 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)

        self.rnn_gyr_1 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)
        self.rnn_gyr_2 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)
        self.rnn_gyr_3 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)

        # Readout layer
        self.fc1 = nn.Sequential(
            nn.Linear(int(hidden_dim*6), int(output_dim*3/2))
            ,nn.ReLU()
        )

        self.fc2 = nn.Linear(int(output_dim*3/2), output_dim)
        self.fc = nn.Linear(int(hidden_dim*6), output_dim)

        self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # x: [batch_size, number of signal, number of channel/axis]
        x = x.view(x.shape[0], 6, -1)

        # acceleration channel 1
        acc_1 = x[:, 0, :]
        # acc_1 = self.init_fc_1(acc_1)
        acc_1 = acc_1.unsqueeze(2)
        acc_1 = torch.reshape(acc_1, [acc_1.shape[0], int(acc_1.shape[1]/self.input_dim), self.input_dim])
        h_acc_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_acc_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_acc_1, (h_acc_n1, c_acc_n1) = self.rnn_acc_1(acc_1, (h_acc_1.detach(), c_acc_1.detach()))
        # embedded_acc_1, _ = self.rnn_acc_1(acc_1)  # , (h_acc_1.detach(), c_acc_1.detach()))
        # embedded_acc_1 = torch.reshape(embedded_acc_1, [batch_size, -1])
        embedded_acc_1 = embedded_acc_1[:, acc_1.shape[1]-1, :]

        # acceleration channel 2
        acc_2 = x[:, 1, :]
        # acc_2 = self.init_fc_2(acc_2)
        acc_2 = acc_2.unsqueeze(2)
        acc_2 = torch.reshape(acc_2, [acc_2.shape[0], int(acc_2.shape[1] / self.input_dim), self.input_dim])
        h_acc_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_acc_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_acc_2, (h_acc_n2, c_acc_n2) = self.rnn_acc_2(acc_2, (h_acc_2.detach(), c_acc_2.detach()))
        # embedded_acc_2, _ = self.rnn_acc_2(acc_2)
        # embedded_acc_2 = torch.reshape(embedded_acc_2, [batch_size, -1])
        embedded_acc_2 = embedded_acc_2[:, acc_1.shape[1] - 1, :]

        # acceleration channel 3
        acc_3 = x[:, 2, :]
        # acc_3 = self.init_fc_3(acc_3)
        acc_3 = acc_3.unsqueeze(2)
        acc_3 = torch.reshape(acc_3, [acc_3.shape[0], int(acc_3.shape[1] / self.input_dim), self.input_dim])
        h_acc_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_acc_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_acc_3, (h_acc_n3, c_acc_n3) = self.rnn_acc_3(acc_3, (h_acc_3.detach(), c_acc_3.detach()))
        # embedded_acc_3, _ = self.rnn_acc_3(acc_3)
        # embedded_acc_3 = torch.reshape(embedded_acc_3, [batch_size, -1])
        embedded_acc_3 = embedded_acc_3[:, acc_1.shape[1] - 1, :]

        # gyroscope channel 1
        gyr_1 = x[:, 3, :]
        # gyr_1 = self.init_fc_4(gyr_1)
        gyr_1 = gyr_1.unsqueeze(2)
        gyr_1 = torch.reshape(gyr_1, [gyr_1.shape[0], int(gyr_1.shape[1] / self.input_dim), self.input_dim])
        h_gyr_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_gyr_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_gyr_1, (h_gyr_n1, c_gyr_n1) = self.rnn_gyr_1(gyr_1, (h_gyr_1.detach(), c_gyr_1.detach()))
        # embedded_gyr_1, _ = self.rnn_gyr_1(gyr_1)
        # embedded_gyr_1 = torch.reshape(embedded_gyr_1, [batch_size, -1])
        embedded_gyr_1 = embedded_gyr_1[:, acc_1.shape[1] - 1, :]

        # gyroscope channel 2
        gyr_2 = x[:, 4, :]
        # gyr_2 = self.init_fc_5(gyr_2)
        gyr_2 = gyr_2.unsqueeze(2)
        gyr_2 = torch.reshape(gyr_2, [gyr_2.shape[0], int(gyr_2.shape[1] / self.input_dim), self.input_dim])
        h_gyr_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_gyr_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_gyr_2, (h_gyr_n2, c_gyr_n2) = self.rnn_gyr_2(gyr_2, (h_gyr_2.detach(), c_gyr_2.detach()))
        # embedded_gyr_2, _ = self.rnn_gyr_2(gyr_2)
        # embedded_gyr_2 = torch.reshape(embedded_gyr_2, [batch_size, -1])
        embedded_gyr_2 = embedded_gyr_2[:, acc_1.shape[1] - 1, :]

        # gyroscope channel 3
        gyr_3 = x[:, 5, :]
        # gyr_3 = self.init_fc_6(gyr_3)
        gyr_3 = gyr_3.unsqueeze(2)
        gyr_3 = torch.reshape(gyr_3, [gyr_3.shape[0], int(gyr_3.shape[1] / self.input_dim), self.input_dim])
        h_gyr_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_gyr_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_gyr_3, (h_gyr_n3, c_gyr_n3) = self.rnn_gyr_3(gyr_3, (h_gyr_3.detach(), c_gyr_3.detach()))
        # embedded_gyr_3, _ = self.rnn_gyr_3(gyr_3)
        # embedded_gyr_3 = torch.reshape(embedded_gyr_3, [batch_size, -1])
        embedded_gyr_3 = embedded_gyr_3[:, acc_1.shape[1] - 1, :]

        template = torch.cat([embedded_acc_1, embedded_acc_2, embedded_acc_3, embedded_gyr_1, embedded_gyr_2, embedded_gyr_3], dim=1)
        # template = torch.cat([embedded_gyr_1, embedded_gyr_2, embedded_gyr_3], dim=1)
        template = self.drop(template)
        if (0):
            out = self.fc1(template)
            out = self.fc2(out)
        else:
            out = self.fc(template)

        out = F.log_softmax(out, 1)
        return out, template


class LSTMSeparatedMtO_6Chan_WHU_80(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, cycle_length=100):
        super(LSTMSeparatedMtO_6Chan_WHU_80, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim

        self.init_fc_1 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            # , nn.ReLU()
        )
        self.init_fc_2 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            # , nn.ReLU()
        )
        self.init_fc_3 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            # , nn.ReLU()
        )
        self.init_fc_4 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            # , nn.ReLU()
        )
        self.init_fc_5 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            # , nn.ReLU()
        )
        self.init_fc_6 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            # , nn.ReLU()
        )
        rate = 0.5
        self.rnn_acc_1 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate,  batch_first=True)
        self.rnn_acc_2 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)
        self.rnn_acc_3 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)

        self.rnn_gyr_1 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)
        self.rnn_gyr_2 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)
        self.rnn_gyr_3 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)

        # Readout layer
        self.fc1 = nn.Sequential(
            nn.Linear(int(hidden_dim*6), int(output_dim*3/2))
            ,nn.ReLU())

        self.fc2 = nn.Linear(int(output_dim*3/2), output_dim)
        self.fc = nn.Linear(int(hidden_dim*6), output_dim)

        self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # x: [batch_size, number of signal, number of channel/axis]
        x = x.view(x.shape[0], 6, -1)

        # acceleration channel 1
        acc_1 = x[:, 0, :]
        # acc_1 = self.init_fc_1(acc_1)
        acc_1 = acc_1.unsqueeze(2)
        acc_1 = torch.reshape(acc_1, [acc_1.shape[0], int(acc_1.shape[1]/self.input_dim), self.input_dim])
        h_acc_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_acc_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_acc_1, (h_acc_n1, c_acc_n1) = self.rnn_acc_1(acc_1, (h_acc_1.detach(), c_acc_1.detach()))
        # embedded_acc_1, _ = self.rnn_acc_1(acc_1)  # , (h_acc_1.detach(), c_acc_1.detach()))
        # embedded_acc_1 = torch.reshape(embedded_acc_1, [batch_size, -1])
        embedded_acc_1 = embedded_acc_1[:, acc_1.shape[1]-1, :]

        # acceleration channel 2
        acc_2 = x[:, 1, :]
        # acc_2 = self.init_fc_2(acc_2)
        acc_2 = acc_2.unsqueeze(2)
        acc_2 = torch.reshape(acc_2, [acc_2.shape[0], int(acc_2.shape[1] / self.input_dim), self.input_dim])
        h_acc_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_acc_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_acc_2, (h_acc_n2, c_acc_n2) = self.rnn_acc_2(acc_2, (h_acc_2.detach(), c_acc_2.detach()))
        # embedded_acc_2, _ = self.rnn_acc_2(acc_2)
        # embedded_acc_2 = torch.reshape(embedded_acc_2, [batch_size, -1])
        embedded_acc_2 = embedded_acc_2[:, acc_1.shape[1] - 1, :]

        # acceleration channel 3
        acc_3 = x[:, 2, :]
        # acc_3 = self.init_fc_3(acc_3)
        acc_3 = acc_3.unsqueeze(2)
        acc_3 = torch.reshape(acc_3, [acc_3.shape[0], int(acc_3.shape[1] / self.input_dim), self.input_dim])
        h_acc_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_acc_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_acc_3, (h_acc_n3, c_acc_n3) = self.rnn_acc_3(acc_3, (h_acc_3.detach(), c_acc_3.detach()))
        # embedded_acc_3, _ = self.rnn_acc_3(acc_3)
        # embedded_acc_3 = torch.reshape(embedded_acc_3, [batch_size, -1])
        embedded_acc_3 = embedded_acc_3[:, acc_1.shape[1] - 1, :]

        # gyroscope channel 1
        gyr_1 = x[:, 3, :]
        # gyr_1 = self.init_fc_4(gyr_1)
        gyr_1 = gyr_1.unsqueeze(2)
        gyr_1 = torch.reshape(gyr_1, [gyr_1.shape[0], int(gyr_1.shape[1] / self.input_dim), self.input_dim])
        h_gyr_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_gyr_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_gyr_1, (h_gyr_n1, c_gyr_n1) = self.rnn_gyr_1(gyr_1, (h_gyr_1.detach(), c_gyr_1.detach()))
        # embedded_gyr_1, _ = self.rnn_gyr_1(gyr_1)
        # embedded_gyr_1 = torch.reshape(embedded_gyr_1, [batch_size, -1])
        embedded_gyr_1 = embedded_gyr_1[:, acc_1.shape[1] - 1, :]

        # gyroscope channel 2
        gyr_2 = x[:, 4, :]
        # gyr_2 = self.init_fc_5(gyr_2)
        gyr_2 = gyr_2.unsqueeze(2)
        gyr_2 = torch.reshape(gyr_2, [gyr_2.shape[0], int(gyr_2.shape[1] / self.input_dim), self.input_dim])
        h_gyr_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_gyr_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_gyr_2, (h_gyr_n2, c_gyr_n2) = self.rnn_gyr_2(gyr_2, (h_gyr_2.detach(), c_gyr_2.detach()))
        # embedded_gyr_2, _ = self.rnn_gyr_2(gyr_2)
        # embedded_gyr_2 = torch.reshape(embedded_gyr_2, [batch_size, -1])
        embedded_gyr_2 = embedded_gyr_2[:, acc_1.shape[1] - 1, :]

        # gyroscope channel 3
        gyr_3 = x[:, 5, :]
        # gyr_3 = self.init_fc_6(gyr_3)
        gyr_3 = gyr_3.unsqueeze(2)
        gyr_3 = torch.reshape(gyr_3, [gyr_3.shape[0], int(gyr_3.shape[1] / self.input_dim), self.input_dim])
        h_gyr_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_gyr_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_gyr_3, (h_gyr_n3, c_gyr_n3) = self.rnn_gyr_3(gyr_3, (h_gyr_3.detach(), c_gyr_3.detach()))
        # embedded_gyr_3, _ = self.rnn_gyr_3(gyr_3)
        # embedded_gyr_3 = torch.reshape(embedded_gyr_3, [batch_size, -1])
        embedded_gyr_3 = embedded_gyr_3[:, acc_1.shape[1] - 1, :]

        template = torch.cat([embedded_acc_1, embedded_acc_2, embedded_acc_3, embedded_gyr_1, embedded_gyr_2, embedded_gyr_3], dim=1)
        # template = torch.cat([embedded_gyr_1, embedded_gyr_2, embedded_gyr_3], dim=1)
        template = self.drop(template)
        if (0):
            out = self.fc1(template)
            out = self.fc2(out)
        else:
            out = self.fc(template)

        # out = F.log_softmax(out, 1)
        return out, template


class LSTMSeparatedMtO_6Chan_WHU_TL(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device, cycle_length=100, embedding_dim=512):
        super(LSTMSeparatedMtO_6Chan_WHU_TL, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim

        self.init_fc_1 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            # , nn.ReLU()
        )
        self.init_fc_2 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            # , nn.ReLU()
        )
        self.init_fc_3 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            # , nn.ReLU()
        )
        self.init_fc_4 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            # , nn.ReLU()
        )
        self.init_fc_5 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            # , nn.ReLU()
        )
        self.init_fc_6 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            # , nn.ReLU()
        )
        rate = 0.5
        self.rnn_acc_1 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)
        self.rnn_acc_2 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)
        self.rnn_acc_3 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)

        self.rnn_gyr_1 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)
        self.rnn_gyr_2 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)
        self.rnn_gyr_3 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)

        # Readout layer
        self.fc1 = nn.Sequential(
            nn.Linear(int(hidden_dim*6), int(output_dim*3/2))
            ,nn.ReLU())

        self.fc2 = nn.Linear(int(output_dim*3/2), output_dim)
        self.fc = nn.Linear(int(hidden_dim*6), output_dim)

        self.drop = nn.Dropout2d(0.5)

        self.fcE = nn.Linear(int(hidden_dim*6), embedding_dim)
        self.device = device

    def forward(self, x):

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # x: [batch_size, number of signal, number of channel/axis]
        x = x.view(x.shape[0], 6, -1)
        # self.rnn.flatten_parameters()

        # acceleration channel 1
        acc_1 = x[:, 0, :]
        # acc_1 = self.init_fc_1(acc_1)
        acc_1 = acc_1.unsqueeze(2)
        acc_1 = torch.reshape(acc_1, [acc_1.shape[0], int(acc_1.shape[1]/self.input_dim), self.input_dim])
        h_acc_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        c_acc_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        # self.rnn_acc_1.flatten_parameters()
        embedded_acc_1, (h_acc_n1, c_acc_n1) = self.rnn_acc_1(acc_1, (h_acc_1.detach(), c_acc_1.detach()))
        # embedded_acc_1, _ = self.rnn_acc_1(acc_1)  # , (h_acc_1.detach(), c_acc_1.detach()))
        # embedded_acc_1 = torch.reshape(embedded_acc_1, [batch_size, -1])
        embedded_acc_1 = embedded_acc_1[:, acc_1.shape[1]-1, :]

        # acceleration channel 2
        acc_2 = x[:, 1, :]
        # acc_2 = self.init_fc_2(acc_2)
        acc_2 = acc_2.unsqueeze(2)
        acc_2 = torch.reshape(acc_2, [acc_2.shape[0], int(acc_2.shape[1] / self.input_dim), self.input_dim])
        h_acc_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        c_acc_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        # self.rnn_acc_2.flatten_parameters()
        embedded_acc_2, (h_acc_n2, c_acc_n2) = self.rnn_acc_2(acc_2, (h_acc_2.detach(), c_acc_2.detach()))
        # embedded_acc_2, _ = self.rnn_acc_2(acc_2)
        # embedded_acc_2 = torch.reshape(embedded_acc_2, [batch_size, -1])
        embedded_acc_2 = embedded_acc_2[:, acc_1.shape[1] - 1, :]

        # acceleration channel 3
        acc_3 = x[:, 2, :]
        # acc_3 = self.init_fc_3(acc_3)
        acc_3 = acc_3.unsqueeze(2)
        acc_3 = torch.reshape(acc_3, [acc_3.shape[0], int(acc_3.shape[1] / self.input_dim), self.input_dim])
        h_acc_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        c_acc_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        #self.rnn_acc_3.flatten_parameters()
        embedded_acc_3, (h_acc_n3, c_acc_n3) = self.rnn_acc_3(acc_3, (h_acc_3.detach(), c_acc_3.detach()))
        # embedded_acc_3, _ = self.rnn_acc_3(acc_3)
        # embedded_acc_3 = torch.reshape(embedded_acc_3, [batch_size, -1])
        embedded_acc_3 = embedded_acc_3[:, acc_1.shape[1] - 1, :]

        # gyroscope channel 1
        gyr_1 = x[:, 3, :]
        # gyr_1 = self.init_fc_4(gyr_1)
        gyr_1 = gyr_1.unsqueeze(2)
        gyr_1 = torch.reshape(gyr_1, [gyr_1.shape[0], int(gyr_1.shape[1] / self.input_dim), self.input_dim])
        h_gyr_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        c_gyr_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        #self.rnn_gyr_1.flatten_parameters()
        embedded_gyr_1, (h_gyr_n1, c_gyr_n1) = self.rnn_gyr_1(gyr_1, (h_gyr_1.detach(), c_gyr_1.detach()))
        # embedded_gyr_1, _ = self.rnn_gyr_1(gyr_1)
        # embedded_gyr_1 = torch.reshape(embedded_gyr_1, [batch_size, -1])
        embedded_gyr_1 = embedded_gyr_1[:, acc_1.shape[1] - 1, :]

        # gyroscope channel 2
        gyr_2 = x[:, 4, :]
        # gyr_2 = self.init_fc_5(gyr_2)
        gyr_2 = gyr_2.unsqueeze(2)
        gyr_2 = torch.reshape(gyr_2, [gyr_2.shape[0], int(gyr_2.shape[1] / self.input_dim), self.input_dim])
        h_gyr_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        c_gyr_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        # self.rnn_gyr_2.flatten_parameters()
        embedded_gyr_2, (h_gyr_n2, c_gyr_n2) = self.rnn_gyr_2(gyr_2, (h_gyr_2.detach(), c_gyr_2.detach()))
        # embedded_gyr_2, _ = self.rnn_gyr_2(gyr_2)
        # embedded_gyr_2 = torch.reshape(embedded_gyr_2, [batch_size, -1])
        embedded_gyr_2 = embedded_gyr_2[:, acc_1.shape[1] - 1, :]

        # gyroscope channel 3
        gyr_3 = x[:, 5, :]
        # gyr_3 = self.init_fc_6(gyr_3)
        gyr_3 = gyr_3.unsqueeze(2)
        gyr_3 = torch.reshape(gyr_3, [gyr_3.shape[0], int(gyr_3.shape[1] / self.input_dim), self.input_dim])
        h_gyr_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        c_gyr_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        # self.rnn_gyr_3.flatten_parameters()
        embedded_gyr_3, (h_gyr_n3, c_gyr_n3) = self.rnn_gyr_3(gyr_3, (h_gyr_3.detach(), c_gyr_3.detach()))
        # embedded_gyr_3, _ = self.rnn_gyr_3(gyr_3)
        # embedded_gyr_3 = torch.reshape(embedded_gyr_3, [batch_size, -1])
        embedded_gyr_3 = embedded_gyr_3[:, acc_1.shape[1] - 1, :]

        template = torch.cat([embedded_acc_1, embedded_acc_2, embedded_acc_3, embedded_gyr_1, embedded_gyr_2, embedded_gyr_3], dim=1)
        # template = torch.cat([embedded_gyr_1, embedded_gyr_2, embedded_gyr_3], dim=1)
        template = self.drop(template)
        if (0):
            template = self.fc1(template)
            out = self.fc2(template)
        else:
            out = self.fcE(template)

        return out


class LSTMSeparatedMtO_allChan_WHU(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, cycle_length=100):
        super(LSTMSeparatedMtO_allChan_WHU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim


        rate = 0.5
        self.rnn_acc_1 = nn.LSTM(6, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)

        # Readout layer
        self.fc1 = nn.Sequential(
            nn.Linear(int(hidden_dim), int(output_dim*3/2))
            ,nn.ReLU())

        self.fc2 = nn.Linear(int(output_dim*3/2), output_dim)
        self.fc = nn.Linear(int(hidden_dim), output_dim)

        self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # x: [batch_size, number of signal, number of channel/axis]
        x = x.view(x.shape[0], 6, -1)

        # acceleration channel 1
        acc_1 = x[:, 0, :]
        # acc_1 = self.init_fc_1(acc_1)
        acc_1 = acc_1.unsqueeze(2)
        acc_1 = torch.reshape(acc_1, [acc_1.shape[0], int(acc_1.shape[1]/self.input_dim), self.input_dim])
        h_acc_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_acc_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_acc_1, (h_acc_n1, c_acc_n1) = self.rnn_acc_1(acc_1, (h_acc_1.detach(), c_acc_1.detach()))

        embedded_acc_1 = embedded_acc_1[:, acc_1.shape[1]-1, :]

        # template = torch.cat([embedded_gyr_1, embedded_gyr_2, embedded_gyr_3], dim=1)
        template = self.drop(embedded_acc_1)
        if (0):
            template = self.fc1(template)
            out = self.fc2(template)
        else:
            out = self.fc(template)

        # out = F.log_softmax(out, 1)
        return out, template


class LSTMSeparatedMtO_6Chan_WHU(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, cycle_length=100):
        super(LSTMSeparatedMtO_6Chan_WHU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim

        self.init_fc_1 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            # , nn.ReLU()
        )
        self.init_fc_2 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            # , nn.ReLU()
        )
        self.init_fc_3 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            # , nn.ReLU()
        )
        self.init_fc_4 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            # , nn.ReLU()
        )
        self.init_fc_5 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            # , nn.ReLU()
        )
        self.init_fc_6 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            # , nn.ReLU()
        )
        rate = 0.5
        self.rnn_acc_1 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)
        self.rnn_acc_2 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)
        self.rnn_acc_3 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)

        self.rnn_gyr_1 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)
        self.rnn_gyr_2 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)
        self.rnn_gyr_3 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, dropout=rate, batch_first=True)

        # Readout layer
        self.fc1 = nn.Sequential(
            nn.Linear(int(hidden_dim*6), int(output_dim*3/2))
            ,nn.ReLU())

        self.fc2 = nn.Linear(int(output_dim*3/2), output_dim)
        self.fc = nn.Linear(int(hidden_dim*6), output_dim)

        self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # x: [batch_size, number of signal, number of channel/axis]
        x = x.view(x.shape[0], 6, -1)

        # acceleration channel 1
        acc_1 = x[:, 0, :]
        # acc_1 = self.init_fc_1(acc_1)
        acc_1 = acc_1.unsqueeze(2)
        acc_1 = torch.reshape(acc_1, [acc_1.shape[0], int(acc_1.shape[1]/self.input_dim), self.input_dim])
        h_acc_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_acc_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_acc_1, (h_acc_n1, c_acc_n1) = self.rnn_acc_1(acc_1, (h_acc_1.detach(), c_acc_1.detach()))
        # embedded_acc_1, _ = self.rnn_acc_1(acc_1)  # , (h_acc_1.detach(), c_acc_1.detach()))
        # embedded_acc_1 = torch.reshape(embedded_acc_1, [batch_size, -1])
        embedded_acc_1 = embedded_acc_1[:, acc_1.shape[1]-1, :]

        # acceleration channel 2
        acc_2 = x[:, 1, :]
        # acc_2 = self.init_fc_2(acc_2)
        acc_2 = acc_2.unsqueeze(2)
        acc_2 = torch.reshape(acc_2, [acc_2.shape[0], int(acc_2.shape[1] / self.input_dim), self.input_dim])
        h_acc_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_acc_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_acc_2, (h_acc_n2, c_acc_n2) = self.rnn_acc_2(acc_2, (h_acc_2.detach(), c_acc_2.detach()))
        # embedded_acc_2, _ = self.rnn_acc_2(acc_2)
        # embedded_acc_2 = torch.reshape(embedded_acc_2, [batch_size, -1])
        embedded_acc_2 = embedded_acc_2[:, acc_1.shape[1] - 1, :]

        # acceleration channel 3
        acc_3 = x[:, 2, :]
        # acc_3 = self.init_fc_3(acc_3)
        acc_3 = acc_3.unsqueeze(2)
        acc_3 = torch.reshape(acc_3, [acc_3.shape[0], int(acc_3.shape[1] / self.input_dim), self.input_dim])
        h_acc_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_acc_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_acc_3, (h_acc_n3, c_acc_n3) = self.rnn_acc_3(acc_3, (h_acc_3.detach(), c_acc_3.detach()))
        # embedded_acc_3, _ = self.rnn_acc_3(acc_3)
        # embedded_acc_3 = torch.reshape(embedded_acc_3, [batch_size, -1])
        embedded_acc_3 = embedded_acc_3[:, acc_1.shape[1] - 1, :]

        # gyroscope channel 1
        gyr_1 = x[:, 3, :]
        # gyr_1 = self.init_fc_4(gyr_1)
        gyr_1 = gyr_1.unsqueeze(2)
        gyr_1 = torch.reshape(gyr_1, [gyr_1.shape[0], int(gyr_1.shape[1] / self.input_dim), self.input_dim])
        h_gyr_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_gyr_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_gyr_1, (h_gyr_n1, c_gyr_n1) = self.rnn_gyr_1(gyr_1, (h_gyr_1.detach(), c_gyr_1.detach()))
        # embedded_gyr_1, _ = self.rnn_gyr_1(gyr_1)
        # embedded_gyr_1 = torch.reshape(embedded_gyr_1, [batch_size, -1])
        embedded_gyr_1 = embedded_gyr_1[:, acc_1.shape[1] - 1, :]

        # gyroscope channel 2
        gyr_2 = x[:, 4, :]
        # gyr_2 = self.init_fc_5(gyr_2)
        gyr_2 = gyr_2.unsqueeze(2)
        gyr_2 = torch.reshape(gyr_2, [gyr_2.shape[0], int(gyr_2.shape[1] / self.input_dim), self.input_dim])
        h_gyr_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_gyr_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_gyr_2, (h_gyr_n2, c_gyr_n2) = self.rnn_gyr_2(gyr_2, (h_gyr_2.detach(), c_gyr_2.detach()))
        # embedded_gyr_2, _ = self.rnn_gyr_2(gyr_2)
        # embedded_gyr_2 = torch.reshape(embedded_gyr_2, [batch_size, -1])
        embedded_gyr_2 = embedded_gyr_2[:, acc_1.shape[1] - 1, :]

        # gyroscope channel 3
        gyr_3 = x[:, 5, :]
        # gyr_3 = self.init_fc_6(gyr_3)
        gyr_3 = gyr_3.unsqueeze(2)
        gyr_3 = torch.reshape(gyr_3, [gyr_3.shape[0], int(gyr_3.shape[1] / self.input_dim), self.input_dim])
        h_gyr_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_gyr_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_gyr_3, (h_gyr_n3, c_gyr_n3) = self.rnn_gyr_3(gyr_3, (h_gyr_3.detach(), c_gyr_3.detach()))
        # embedded_gyr_3, _ = self.rnn_gyr_3(gyr_3)
        # embedded_gyr_3 = torch.reshape(embedded_gyr_3, [batch_size, -1])
        embedded_gyr_3 = embedded_gyr_3[:, acc_1.shape[1] - 1, :]

        template = torch.cat([embedded_acc_1, embedded_acc_2, embedded_acc_3, embedded_gyr_1, embedded_gyr_2, embedded_gyr_3], dim=1)
        # template = torch.cat([embedded_gyr_1, embedded_gyr_2, embedded_gyr_3], dim=1)
        template = self.drop(template)
        if (0):
            template = self.fc1(template)
            out = self.fc2(template)
        else:
            out = self.fc(template)

        out = F.log_softmax(out, 1)
        return out, template


class Gait_Recognition_OU_Segment_100_6_adjust(torch.nn.Module):

    def __init__(self, usernumber):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Gait_Recognition_OU_Segment_100_6_adjust, self).__init__()

        self.user_number = usernumber

        self.conv1_acc = nn.Sequential(
            nn.Conv2d(1, 240, kernel_size=(1, 9), stride=(1, 1), padding=(0, 0))
            , nn.ReLU()
            , nn.BatchNorm2d(240)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv1_gyr = nn.Sequential(
            nn.Conv2d(1, 240, kernel_size=(1, 9), stride=(1, 1), padding=(0, 0))
            , nn.ReLU()
            , nn.BatchNorm2d(240)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv2 = nn.Sequential(
            nn.Conv2d(240, 300, kernel_size=(1, 7), stride=(1, 1), padding=(0, 1))
            , nn.ReLU()
            , nn.BatchNorm2d(300)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv3 = nn.Sequential(
            nn.Conv2d(300, 360, kernel_size=(1, 5), stride=(1, 1), padding=(0, 1))
            , nn.ReLU()
            , nn.BatchNorm2d(360)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv4 = nn.Sequential(
            nn.Conv2d(360, 420, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0))
            , nn.ReLU()
            , nn.BatchNorm2d(420)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.avgpool = nn.Sequential(
            nn.AvgPool2d((1, 4), stride=(1, 4))
            , nn.Dropout2d(0.5)
        )

        self.fc1 = nn.Sequential(
             nn.Linear(3780, int(self.user_number *3/2))
             , nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(int(self.user_number*3/2) , self.user_number)
        )

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        # input x : 23 x 59049 x 1
        # expected conv1d input : minibatch_size x num_channel x width
        batch_size = x.size(0)
        x = x.view(x.shape[0], 1, 8, 120)

        gyr_data = x[:, :, 0:3, :]
        acc_data = x[:, :, 4:7, :]
        # x : batch_size x 1 x 900

        out_gyr = self.conv1_gyr(gyr_data)
        out_acc = self.conv1_acc(acc_data)
        out = torch.cat([out_gyr, out_acc], dim=3)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.avgpool(out)

        out = out.view(batch_size, -1)  # flatten the tensor
        embedded = self.fc1(out)
        out = self.fc2(embedded)
        out = F.log_softmax(out, 1)
        return out, embedded




class Gait_Recognition_WHU_Segment_80_6(torch.nn.Module):

    def __init__(self, usernumber):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Gait_Recognition_WHU_Segment_80_6, self).__init__()

        self.user_number = usernumber

        self.conv1_acc = nn.Sequential(
            nn.Conv2d(1, 240, kernel_size=(1, 9), stride=(1, 1), padding=(0, 0))
            , nn.ReLU()
            , nn.BatchNorm2d(240)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv1_gyr = nn.Sequential(
            nn.Conv2d(1, 240, kernel_size=(1, 9), stride=(1, 1), padding=(0, 0))
            , nn.ReLU()
            , nn.BatchNorm2d(240)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv2 = nn.Sequential(
            nn.Conv2d(240, 300, kernel_size=(1, 7), stride=(1, 1), padding=(0, 1))
            , nn.ReLU()
            , nn.BatchNorm2d(300)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv3 = nn.Sequential(
            nn.Conv2d(300, 360, kernel_size=(1, 5), stride=(1, 1), padding=(0, 1))
            , nn.ReLU()
            , nn.BatchNorm2d(360)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv4 = nn.Sequential(
            nn.Conv2d(360, 420, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
            , nn.ReLU()
            , nn.BatchNorm2d(420)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.avgpool = nn.Sequential(
            nn.AvgPool2d((1, 4), stride=(1, 4))
            , nn.Dropout2d(0.5)
        )

        self.fc1 = nn.Sequential(
             nn.Linear(2520, int(self.user_number *3/2))
             , nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(int(self.user_number*3/2) , self.user_number)
            )

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        # input x : 23 x 59049 x 1
        # expected conv1d input : minibatch_size x num_channel x width
        batch_size = x.size(0)
        x = x.view(x.shape[0], 1, 6, -1)

        gyr_data = x[:, :, 0:3, :]
        acc_data = x[:, :, 3:6, :]
        # x : batch_size x 1 x 900

        out_gyr = self.conv1_gyr(gyr_data)
        out_acc = self.conv1_acc(acc_data)
        out = torch.cat([out_gyr, out_acc], dim=3)
        out = self.conv2(out)

        out = self.conv3(out)
        out = self.conv4(out)
        # embedded = out.view(batch_size, -1)
        out = self.avgpool(out)

        out = out.view(batch_size, -1)  # flatten the tensor
        out = self.fc1(out)
        embedded = out
        out = self.fc2(out)
        """
        standard
        out_gyr = self.conv1_gyr(gyr_data)
        out_acc = self.conv1_acc(acc_data)
        out = torch.cat([out_gyr, out_acc], dim=3)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.avgpool(out)

        embedded = out.view(batch_size, -1)  # flatten the tensor
        embedded = self.fc1(embedded)
        out = self.fc2(embedded)
        """
        out = F.log_softmax(out, 1)
        return out, embedded

class Gait_Recognition_WHU_Segment_80_61(torch.nn.Module):

    def __init__(self, usernumber):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Gait_Recognition_WHU_Segment_80_61, self).__init__()

        self.user_number = usernumber

        self.conv1_acc = nn.Sequential(
            nn.Conv2d(1, 240, kernel_size=(1, 9), stride=(1, 1), padding=(0, 0))
            , nn.ReLU()
            , nn.BatchNorm2d(240)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv1_gyr = nn.Sequential(
            nn.Conv2d(1, 240, kernel_size=(1, 9), stride=(1, 1), padding=(0, 0))
            , nn.ReLU()
            , nn.BatchNorm2d(240)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv2 = nn.Sequential(
            nn.Conv2d(240, 300, kernel_size=(1, 7), stride=(1, 1), padding=(0, 1))
            , nn.ReLU()
            , nn.BatchNorm2d(300)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv3 = nn.Sequential(
            nn.Conv2d(300, 360, kernel_size=(1, 5), stride=(1, 1), padding=(0, 1))
            , nn.ReLU()
            , nn.BatchNorm2d(360)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv4 = nn.Sequential(
            nn.Conv2d(360, 420, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
            , nn.ReLU()
            , nn.BatchNorm2d(420)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.avgpool = nn.Sequential(
            nn.AvgPool2d((1, 4), stride=(1, 4))
            , nn.Dropout2d(0.5)
        )

        self.fc1 = nn.Sequential(
             nn.Linear(2520, int(self.user_number *3/2))
             , nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(int(self.user_number*3/2) , self.user_number)
            )

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        # input x : 23 x 59049 x 1
        # expected conv1d input : minibatch_size x num_channel x width
        batch_size = x.size(0)
        x = x.view(x.shape[0], 1, 6, -1)

        gyr_data = x[:, :, 0:3, :]
        acc_data = x[:, :, 3:6, :]
        # x : batch_size x 1 x 900

        out_gyr = self.conv1_gyr(gyr_data)
        out_acc = self.conv1_acc(acc_data)
        out = torch.cat([out_gyr, out_acc], dim=3)
        out = self.conv2(out)

        out = self.conv3(out)
        out = self.conv4(out)
        # embedded = out.view(batch_size, -1)
        out = self.avgpool(out)

        out = out.view(batch_size, -1)  # flatten the tensor
        embedded = out
        out = self.fc1(out)

        out = self.fc2(out)
        """
        standard
        out_gyr = self.conv1_gyr(gyr_data)
        out_acc = self.conv1_acc(acc_data)
        out = torch.cat([out_gyr, out_acc], dim=3)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.avgpool(out)

        embedded = out.view(batch_size, -1)  # flatten the tensor
        embedded = self.fc1(embedded)
        out = self.fc2(embedded)
        """
        out = F.log_softmax(out, 1)
        return out, embedded


class Gait_Recognition_OU_Segment_100_6_TL(torch.nn.Module):

    def __init__(self, usernumber, embedding_dim=128):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Gait_Recognition_OU_Segment_100_6_TL, self).__init__()

        self.user_number = usernumber
        self.embedding_dim=embedding_dim

        self.conv1_acc = nn.Sequential(
            nn.Conv2d(1, 240, kernel_size=(1, 9), stride=(1, 1), padding=(0, 0))
            , nn.ReLU()
            , nn.BatchNorm2d(240)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv1_gyr = nn.Sequential(
            nn.Conv2d(1, 240, kernel_size=(1, 9), stride=(1, 1), padding=(0, 0))
            , nn.ReLU()
            , nn.BatchNorm2d(240)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv2 = nn.Sequential(
            nn.Conv2d(240, 300, kernel_size=(1, 7), stride=(1, 1), padding=(0, 1))
            , nn.ReLU()
            , nn.BatchNorm2d(300)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv3 = nn.Sequential(
            nn.Conv2d(300, 360, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0))
            , nn.ReLU()
            , nn.BatchNorm2d(360)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv4 = nn.Sequential(
            nn.Conv2d(360, 420, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
            , nn.ReLU()
            , nn.BatchNorm2d(420)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.avgpool = nn.Sequential(
            nn.AvgPool2d((1, 5), stride=(1, 5))
            , nn.Dropout2d(0.5)
        )

        self.fc1 = nn.Sequential(
             nn.Linear(2520, int(self.user_number *3/2))
             , nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(int(self.user_number*3/2) , self.embedding_dim)
            )

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        # input x : 23 x 59049 x 1
        # expected conv1d input : minibatch_size x num_channel x width
        batch_size = x.size(0)
        x = x.view(x.shape[0], 1, 6, -1)

        gyr_data = x[:, :, 0:3, :]
        acc_data = x[:, :, 3:6, :]
        # x : batch_size x 1 x 900

        out_gyr = self.conv1_gyr(gyr_data)
        out_acc = self.conv1_acc(acc_data)
        out = torch.cat([out_gyr, out_acc], dim=3)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.avgpool(out)

        embedded = out.view(batch_size, -1)  # flatten the tensor
        out = self.fc1(embedded)
        out = self.fc2(out)
        # out = F.log_softmax(out, 1)
        return out


class Gait_Recognition_OU_Segment_100_6(torch.nn.Module):

    def __init__(self, usernumber):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Gait_Recognition_OU_Segment_100_6, self).__init__()

        self.user_number = usernumber

        self.conv1_acc = nn.Sequential(
            nn.Conv2d(1, 240, kernel_size=(1, 9), stride=(1, 1), padding=(0, 0))
            , nn.ReLU()
            , nn.BatchNorm2d(240)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv1_gyr = nn.Sequential(
            nn.Conv2d(1, 240, kernel_size=(1, 9), stride=(1, 1), padding=(0, 0))
            , nn.ReLU()
            , nn.BatchNorm2d(240)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv2 = nn.Sequential(
            nn.Conv2d(240, 300, kernel_size=(1, 7), stride=(1, 1), padding=(0, 1))
            , nn.ReLU()
            , nn.BatchNorm2d(300)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv3 = nn.Sequential(
            nn.Conv2d(300, 360, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0))
            , nn.ReLU()
            , nn.BatchNorm2d(360)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.conv4 = nn.Sequential(
            nn.Conv2d(360, 420, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
            , nn.ReLU()
            , nn.BatchNorm2d(420)
            , nn.MaxPool2d((1, 2), stride=(1, 2)))

        self.avgpool = nn.Sequential(
            nn.AvgPool2d((1, 5), stride=(1, 5))
            , nn.Dropout2d(0.5)
        )

        self.fc1 = nn.Sequential(
             nn.Linear(2520, int(self.user_number *3/2))
             , nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(int(self.user_number*3/2) , self.user_number)
            )

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        # input x : 23 x 59049 x 1
        # expected conv1d input : minibatch_size x num_channel x width
        batch_size = x.size(0)
        x = x.view(x.shape[0], 1, 6, -1)

        gyr_data = x[:, :, 0:3, :]
        acc_data = x[:, :, 3:6, :]
        # x : batch_size x 1 x 900

        out_gyr = self.conv1_gyr(gyr_data)
        out_acc = self.conv1_acc(acc_data)
        out = torch.cat([out_gyr, out_acc], dim=3)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.avgpool(out)

        embedded = out.view(batch_size, -1)  # flatten the tensor
        out = self.fc1(embedded)
        out = self.fc2(out)
        out = F.log_softmax(out, 1)
        return out, embedded


class LSTMSeparatedMtO4D(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, cycle_length=120):
        super(LSTMSeparatedMtO4D, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim

        self.init_fc_1 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            #, nn.ReLU()
        )
        self.init_fc_2 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            #, nn.ReLU()
        )
        self.init_fc_3 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            #, nn.ReLU()
        )
        self.init_fc_4 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            #, nn.ReLU()
        )


        self.rnn_acc_1 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True)
        self.rnn_acc_2 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True)
        self.rnn_acc_3 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True)
        self.rnn_acc_4 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True)


        # Readout layer
        self.fc1 = nn.Linear(int(hidden_dim*4), int(output_dim))
        self.fc2 = nn.Linear(int(output_dim), output_dim)
        self.fc = nn.Linear(int(hidden_dim*4), output_dim)

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # x: [batch_size, number of signal, number of channel/axis]
        x = x.view(x.shape[0], 6, -1)

        scale = self.hidden_dim*120/self.input_dim*4
        batch_size = x.shape[0]

        # acceleration channel 1
        acc_1 = x[:, 0, :]
        acc_1 = self.init_fc_1(acc_1)
        acc_1 = acc_1.unsqueeze(2)
        acc_1 = torch.reshape(acc_1, [acc_1.shape[0], int(acc_1.shape[1]/self.input_dim), self.input_dim])
        h_acc_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_acc_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_acc_1, (h_acc_n1, c_acc_n1) = self.rnn_acc_1(acc_1, (h_acc_1.detach(), c_acc_1.detach()))
        # embedded_acc_1, _ = self.rnn_acc_1(acc_1)  # , (h_acc_1.detach(), c_acc_1.detach()))
        # embedded_acc_1 = torch.reshape(embedded_acc_1, [batch_size, -1])
        embedded_acc_1 = embedded_acc_1[:, acc_1.shape[1]-1, :]

        # acceleration channel 2
        acc_2 = x[:, 1, :]
        acc_2 = self.init_fc_2(acc_2)
        acc_2 = acc_2.unsqueeze(2)
        acc_2 = torch.reshape(acc_2, [acc_2.shape[0], int(acc_2.shape[1] / self.input_dim), self.input_dim])
        h_acc_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_acc_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_acc_2, (h_acc_n2, c_acc_n2) = self.rnn_acc_2(acc_2, (h_acc_2.detach(), c_acc_2.detach()))
        # embedded_acc_2, _ = self.rnn_acc_2(acc_2)
        # embedded_acc_2 = torch.reshape(embedded_acc_2, [batch_size, -1])
        embedded_acc_2 = embedded_acc_2[:, acc_1.shape[1] - 1, :]

        # acceleration channel 3
        acc_3 = x[:, 2, :]
        acc_3 = self.init_fc_3(acc_3)
        acc_3 = acc_3.unsqueeze(2)
        acc_3 = torch.reshape(acc_3, [acc_3.shape[0], int(acc_3.shape[1] / self.input_dim), self.input_dim])
        h_acc_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_acc_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_acc_3, (h_acc_n3, c_acc_n3) = self.rnn_acc_3(acc_3, (h_acc_3.detach(), c_acc_3.detach()))
        # embedded_acc_3, _ = self.rnn_acc_3(acc_3)
        # embedded_acc_3 = torch.reshape(embedded_acc_3, [batch_size, -1])
        embedded_acc_3 = embedded_acc_3[:, acc_1.shape[1] - 1, :]

        # acceleration channel 3
        acc_4 = x[:, 3, :]
        acc_4 = self.init_fc_4(acc_4)
        acc_4 = acc_4.unsqueeze(2)
        acc_4 = torch.reshape(acc_4, [acc_4.shape[0], int(acc_4.shape[1] / self.input_dim), self.input_dim])
        h_acc_4 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_acc_4 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_acc_4, (h_acc_n4, c_acc_n4) = self.rnn_acc_4(acc_4, (h_acc_4.detach(), c_acc_4.detach()))
        # embedded_acc_4, _ = self.rnn_acc_4(acc_4)
        # embedded_acc_4 = torch.reshape(embedded_acc_4, [batch_size, -1])
        embedded_acc_4 = embedded_acc_4[:, acc_1.shape[1] - 1, :]

        template = torch.cat([embedded_acc_1, embedded_acc_2, embedded_acc_3, embedded_acc_4], dim=1)
        #template = torch.cat([embedded_acc_1, embedded_acc_2, embedded_acc_3], dim=1)

        if (1):
            template = self.fc1(template)
            out = self.fc2(template)
        else:
            out = self.fc(template)

        # out = F.log_softmax(out, 1)
        return out, template


class LSTMSeparatedMtO(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, cycle_length=120):
        super(LSTMSeparatedMtO, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim

        self.init_fc_1 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            , nn.ReLU())
        self.init_fc_2 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            , nn.ReLU())
        self.init_fc_3 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            , nn.ReLU())
        self.init_fc_4 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            , nn.ReLU())
        self.init_fc_5 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            , nn.ReLU())
        self.init_fc_6 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            , nn.ReLU())
        self.init_fc_7 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            , nn.ReLU())
        self.init_fc_8 = nn.Sequential(
            nn.Linear(cycle_length, cycle_length)
            , nn.ReLU())

        self.rnn_acc_1 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True)
        self.rnn_acc_2 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True)
        self.rnn_acc_3 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True)
        self.rnn_acc_4 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True)

        self.rnn_gyr_1 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True)
        self.rnn_gyr_2 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True)
        self.rnn_gyr_3 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True)
        self.rnn_gyr_4 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True)

        # Readout layer
        self.fc1 = nn.Linear(int(hidden_dim*6), int(output_dim))
        self.fc2 = nn.Linear(int(output_dim), output_dim)
        self.fc = nn.Linear(int(hidden_dim*6), output_dim)

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # x: [batch_size, number of signal, number of channel/axis]
        scale = self.hidden_dim*120/self.input_dim*6
        batch_size = x.shape[0]

        # acceleration channel 1
        acc_1 = x[:, :, 0]
        acc_1 = self.init_fc_1(acc_1)
        acc_1 = acc_1.unsqueeze(2)
        acc_1 = torch.reshape(acc_1, [acc_1.shape[0], int(acc_1.shape[1]/self.input_dim), self.input_dim])
        h_acc_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_acc_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_acc_1, (h_acc_n1, c_acc_n1) = self.rnn_acc_1(acc_1, (h_acc_1.detach(), c_acc_1.detach()))
        # embedded_acc_1, _ = self.rnn_acc_1(acc_1)  # , (h_acc_1.detach(), c_acc_1.detach()))
        # embedded_acc_1 = torch.reshape(embedded_acc_1, [batch_size, -1])
        embedded_acc_1 = embedded_acc_1[:, acc_1.shape[1]-1, :]

        # acceleration channel 2
        acc_2 = x[:, :, 1]
        acc_2 = self.init_fc_2(acc_2)
        acc_2 = acc_2.unsqueeze(2)
        acc_2 = torch.reshape(acc_2, [acc_2.shape[0], int(acc_2.shape[1] / self.input_dim), self.input_dim])
        h_acc_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_acc_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_acc_2, (h_acc_n2, c_acc_n2) = self.rnn_acc_2(acc_2, (h_acc_2.detach(), c_acc_2.detach()))
        # embedded_acc_2, _ = self.rnn_acc_2(acc_2)
        # embedded_acc_2 = torch.reshape(embedded_acc_2, [batch_size, -1])
        embedded_acc_2 = embedded_acc_2[:, acc_1.shape[1] - 1, :]

        # acceleration channel 3
        acc_3 = x[:, :, 2]
        acc_3 = self.init_fc_3(acc_3)
        acc_3 = acc_3.unsqueeze(2)
        acc_3 = torch.reshape(acc_3, [acc_3.shape[0], int(acc_3.shape[1] / self.input_dim), self.input_dim])
        h_acc_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_acc_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_acc_3, (h_acc_n3, c_acc_n3) = self.rnn_acc_3(acc_3, (h_acc_3.detach(), c_acc_3.detach()))
        # embedded_acc_3, _ = self.rnn_acc_3(acc_3)
        # embedded_acc_3 = torch.reshape(embedded_acc_3, [batch_size, -1])
        embedded_acc_3 = embedded_acc_3[:, acc_1.shape[1] - 1, :]

        # acceleration channel 3
        acc_4 = x[:, :, 3]
        acc_4 = self.init_fc_4(acc_4)
        acc_4 = acc_4.unsqueeze(2)
        acc_4 = torch.reshape(acc_4, [acc_4.shape[0], int(acc_4.shape[1] / self.input_dim), self.input_dim])
        h_acc_4 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_acc_4 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_acc_4, (h_acc_n4, c_acc_n4) = self.rnn_acc_4(acc_4, (h_acc_4.detach(), c_acc_4.detach()))
        # embedded_acc_4, _ = self.rnn_acc_4(acc_4)
        # embedded_acc_4 = torch.reshape(embedded_acc_4, [batch_size, -1])
        embedded_acc_4 = embedded_acc_4[:, acc_1.shape[1] - 1, :]

        # gyroscope channel 1
        gyr_1 = x[:, :, 4]
        gyr_1 = self.init_fc_5(gyr_1)
        gyr_1 = gyr_1.unsqueeze(2)
        gyr_1 = torch.reshape(gyr_1, [gyr_1.shape[0], int(gyr_1.shape[1] / self.input_dim), self.input_dim])
        h_gyr_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_gyr_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_gyr_1, (h_gyr_n1, c_gyr_n1) = self.rnn_gyr_1(gyr_1, (h_gyr_1.detach(), c_gyr_1.detach()))
        # embedded_gyr_1, _ = self.rnn_gyr_1(gyr_1)
        # embedded_gyr_1 = torch.reshape(embedded_gyr_1, [batch_size, -1])
        embedded_gyr_1 = embedded_gyr_1[:, acc_1.shape[1] - 1, :]

        # gyroscope channel 2
        gyr_2 = x[:, :, 5]
        gyr_2 = self.init_fc_6(gyr_2)
        gyr_2 = gyr_2.unsqueeze(2)
        gyr_2 = torch.reshape(gyr_2, [gyr_2.shape[0], int(gyr_2.shape[1] / self.input_dim), self.input_dim])
        h_gyr_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_gyr_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_gyr_2, (h_gyr_n2, c_gyr_n2) = self.rnn_gyr_2(gyr_2, (h_gyr_2.detach(), c_gyr_2.detach()))
        # embedded_gyr_2, _ = self.rnn_gyr_2(gyr_2)
        # embedded_gyr_2 = torch.reshape(embedded_gyr_2, [batch_size, -1])
        embedded_gyr_2 = embedded_gyr_2[:, acc_1.shape[1] - 1, :]

        # gyroscope channel 3
        gyr_3 = x[:, :, 6]
        gyr_3 = self.init_fc_7(gyr_3)
        gyr_3 = gyr_3.unsqueeze(2)
        gyr_3 = torch.reshape(gyr_3, [gyr_3.shape[0], int(gyr_3.shape[1] / self.input_dim), self.input_dim])
        h_gyr_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_gyr_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_gyr_3, (h_gyr_n3, c_gyr_n3) = self.rnn_gyr_3(gyr_3, (h_gyr_3.detach(), c_gyr_3.detach()))
        # embedded_gyr_3, _ = self.rnn_gyr_3(gyr_3)
        # embedded_gyr_3 = torch.reshape(embedded_gyr_3, [batch_size, -1])
        embedded_gyr_3 = embedded_gyr_3[:, acc_1.shape[1] - 1, :]

        # gyroscope channel 4
        gyr_4 = x[:, :, 7]
        gyr_4 = self.init_fc_8(gyr_4)
        gyr_4 = gyr_4.unsqueeze(2)
        gyr_4 = torch.reshape(gyr_4, [gyr_4.shape[0], int(gyr_4.shape[1] / self.input_dim), self.input_dim])
        h_gyr_4 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_gyr_4 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_gyr_4, (h_gyr_n4, c_gyr_n4) = self.rnn_gyr_4(gyr_4, (h_gyr_4.detach(), c_gyr_4.detach()))
        # embedded_gyr_4, _ = self.rnn_gyr_4(gyr_4)
        # embedded_gyr_4 = torch.reshape(embedded_gyr_4, [batch_size, -1])
        embedded_gyr_4 = embedded_gyr_4[:, acc_1.shape[1] - 1, :]
        template = torch.cat([embedded_acc_1, embedded_acc_2, embedded_acc_3, embedded_gyr_1, embedded_gyr_2, embedded_gyr_3], dim=1)
        #template = torch.cat([embedded_acc_1, embedded_acc_2, embedded_acc_3], dim=1)

        if (1):
            template = self.fc1(template)
            out = self.fc2(template)
        else:
            out = self.fc(template)

        # out = F.log_softmax(out, 1)
        return out, template


class LSTMSeparated(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, cycle_length=120):
        super(LSTMSeparated, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim

        self.rnn_acc_1 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True)
        self.rnn_acc_2 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True)
        self.rnn_acc_3 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True)
        self.rnn_acc_4 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True)

        self.rnn_gyr_1 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True)
        self.rnn_gyr_2 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True)
        self.rnn_gyr_3 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True)
        self.rnn_gyr_4 = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True)

        # Readout layer
        self.fc1 = nn.Linear(int(hidden_dim*8*120/input_dim), int(output_dim*3/2))
        self.fc2 = nn.Linear(int(output_dim*3/2), output_dim)

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # x: [batch_size, number of signal, number of channel/axis]
        # scale = self.hidden_dim*120/self.input_dim*6
        batch_size = x.shape[0]

        # acceleration channel 1
        acc_1 = x[:, :, 0]
        acc_1 = acc_1.unsqueeze(2)
        acc_1 = torch.reshape(acc_1, [acc_1.shape[0], int(acc_1.shape[1]/self.input_dim), self.input_dim])
        h_acc_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_acc_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_acc_1, (h_acc_n1, c_acc_n1) = self.rnn_acc_1(acc_1, (h_acc_1.detach(), c_acc_1.detach()))
        embedded_acc_1 = torch.reshape(embedded_acc_1, [batch_size, -1])

        # acceleration channel 2
        acc_2 = x[:, :, 1]
        acc_2 = acc_2.unsqueeze(2)
        acc_2 = torch.reshape(acc_2, [acc_2.shape[0], int(acc_2.shape[1] / self.input_dim), self.input_dim])
        h_acc_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_acc_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_acc_2, (h_acc_n2, c_acc_n2) = self.rnn_acc_2(acc_2, (h_acc_2.detach(), c_acc_2.detach()))
        embedded_acc_2 = torch.reshape(embedded_acc_2, [batch_size, -1])

        # acceleration channel 3
        acc_3 = x[:, :, 2]
        acc_3 = acc_3.unsqueeze(2)
        acc_3 = torch.reshape(acc_3, [acc_3.shape[0], int(acc_3.shape[1] / self.input_dim), self.input_dim])
        h_acc_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_acc_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_acc_3, (h_acc_n3, c_acc_n3) = self.rnn_acc_3(acc_3, (h_acc_3.detach(), c_acc_3.detach()))
        embedded_acc_3 = torch.reshape(embedded_acc_3, [batch_size, -1])

        # acceleration channel 3
        acc_4 = x[:, :, 3]
        acc_4 = acc_4.unsqueeze(2)
        acc_4 = torch.reshape(acc_4, [acc_4.shape[0], int(acc_4.shape[1] / self.input_dim), self.input_dim])
        h_acc_4 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_acc_4 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_acc_4, (h_acc_n4, c_acc_n4) = self.rnn_acc_4(acc_4, (h_acc_4.detach(), c_acc_4.detach()))
        embedded_acc_4 = torch.reshape(embedded_acc_4, [batch_size, -1])

        # gyroscope channel 1
        gyr_1 = x[:, :, 4]
        gyr_1 = gyr_1.unsqueeze(2)
        gyr_1 = torch.reshape(gyr_1, [gyr_1.shape[0], int(gyr_1.shape[1] / self.input_dim), self.input_dim])
        h_gyr_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_gyr_1 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_gyr_1, (h_gyr_n1, c_gyr_n1) = self.rnn_gyr_1(gyr_1, (h_gyr_1.detach(), c_gyr_1.detach()))
        embedded_gyr_1 = torch.reshape(embedded_gyr_1, [batch_size, -1])

        # gyroscope channel 2
        gyr_2 = x[:, :, 5]
        gyr_2 = gyr_2.unsqueeze(2)
        gyr_2 = torch.reshape(gyr_2, [gyr_2.shape[0], int(gyr_2.shape[1] / self.input_dim), self.input_dim])
        h_gyr_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_gyr_2 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_gyr_2, (h_gyr_n2, c_gyr_n2) = self.rnn_gyr_2(gyr_2, (h_gyr_2.detach(), c_gyr_2.detach()))
        embedded_gyr_2 = torch.reshape(embedded_gyr_2, [batch_size, -1])

        # gyroscope channel 3
        gyr_3 = x[:, :, 6]
        gyr_3 = gyr_3.unsqueeze(2)
        gyr_3 = torch.reshape(gyr_3, [gyr_3.shape[0], int(gyr_3.shape[1] / self.input_dim), self.input_dim])
        h_gyr_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_gyr_3 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_gyr_3, (h_gyr_n3, c_gyr_n3) = self.rnn_gyr_3(gyr_3, (h_gyr_3.detach(), c_gyr_3.detach()))
        embedded_gyr_3 = torch.reshape(embedded_gyr_3, [batch_size, -1])

        # gyroscope channel 4
        gyr_4 = x[:, :, 7]
        gyr_4 = gyr_4.unsqueeze(2)
        gyr_4 = torch.reshape(gyr_4, [gyr_4.shape[0], int(gyr_4.shape[1] / self.input_dim), self.input_dim])
        h_gyr_4 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c_gyr_4 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        embedded_gyr_4, (h_gyr_n4, c_gyr_n4) = self.rnn_gyr_4(gyr_4, (h_gyr_4.detach(), c_gyr_4.detach()))
        embedded_gyr_4 = torch.reshape(embedded_gyr_4, [batch_size, -1])

        embedded_template = torch.cat([embedded_acc_1, embedded_acc_2, embedded_acc_3, embedded_acc_4, embedded_gyr_1, embedded_gyr_2, embedded_gyr_3, embedded_gyr_4], dim=1)
        template = self.fc1(embedded_template)
        out = self.fc2(template)

        out = F.log_softmax(out, 1)
        return out, template


class LSTMModel_FixedLength(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=50, layer_dim=2, output_dim=118, cycle_length=128 ):
        super(LSTMModel_FixedLength, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        self.fc_init = nn.Linear(cycle_length*8, cycle_length*8)
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Initialize hidden state with zeros
        # print(x)
        scale = self.input_dim / x.shape[2]
        temp = x.shape[1]
        # x = torch.reshape(x, (x.shape[0], -1))
        # x = self.fc_init(x)
        # print(y)

        # x = x.view(x.shape[0],  int(x.shape[1]/scale), -1, order=F)
        x = torch.reshape(x, (x.shape[0],  int(temp/scale), -1),)

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        embedded, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(embedded[:, -1, :])
        # out.size() --> 100, 10
        return out, embedded


class LSTMModel1(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, cycle_length=120 ):
        super(LSTMModel1, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        self.fc_init = nn.Linear(cycle_length*8, cycle_length*8)
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Initialize hidden state with zeros
        # print(x)
        scale = self.input_dim / x.shape[2]
        temp = x.shape[1]
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.fc_init(x)
        # print(y)

        # x = x.view(x.shape[0],  int(x.shape[1]/scale), -1, order=F)
        x = torch.reshape(x, (x.shape[0],  int(temp/scale), -1),)

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        embedded, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(embedded[:, -1, :])
        # out.size() --> 100, 10
        return out, embedded


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, cycle_length=120):
        #
        # input_dim:    the number of scalars that is fed to an LSTM node at each stage
        # hidden_dim:   the number of scalars that is returned from LSTM node at each stage
        # layer_dim:    the number of layers in the RNN network
        # output_dim:   should be equal to the number of users

        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.layer_dim = layer_dim
        self.cycle_length = cycle_length

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.length = int(self.cycle_length/input_dim*8)    # the number of inputs for a given gait cycle
        self.fc1 = nn.Linear(hidden_dim*self.length, output_dim*2)
        self.fc = nn.Linear(output_dim*2, output_dim)

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Initialize hidden state with zeros
        scale = self.input_dim/x.shape[2]
        batch_size = x.shape[0]

        x = torch.reshape(x, (x.shape[0],  int(x.shape[1]/scale), -1),).contiguous()

        # initialize the cell state
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        embedded, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        embedded = torch.reshape(embedded, [batch_size, -1])
        embedded = self.fc1(embedded)

        out = F.log_softmax(self.fc(embedded), 1)
        return out, embedded


class Gait_Verifi_RNN(torch.nn.Module):
    def __init__(self, in_feature=8, hidden_feature=100, num_class=10, num_layers=1):
        super(Gait_Verifi_RNN, self).__init__()

        self.rnn = nn.LSTM(in_feature, hidden_feature, num_layers)  # Use two layers of lstm
        self.classifier = nn.Sequential(
            nn.Linear(hidden_feature, num_class)
            , nn.ReLU())

    def forward(self, x):
        batch_size = x.size(0)

        #h0 = torch.zeros(self.number_layer, batch_size, self.hidden_size).requires_grad_()
        #c0 = torch.zeros(self.number_layer, batch_size, self.hidden_size).requires_grad_()
        embedded, _ = self.rnn(x)

        out = self.classifier(embedded[:, -1, :])
        out = F.log_softmax(out, 1)
        return out, embedded


class Gait_Verifi_CNN(torch.nn.Module):

    def __init__(self, user_number):
        super(Gait_Verifi_CNN, self).__init__()

        self.user_number = user_number
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(7, 3), stride=(1, 1), padding=(1, 1))
            , nn.ReLU()
            , nn.MaxPool2d((2, 2), stride=(2, 2))
            , nn.BatchNorm2d(32))

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(7, 3), stride=(1, 1), padding=(0, 1))
            , nn.ReLU()
            , nn.MaxPool2d((2, 2), stride=(2, 2))
            , nn.BatchNorm2d(64))

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=(7, 3), stride=(1, 1), padding=(0, 1))
            , nn.ReLU()
            , nn.MaxPool2d((2, 2), stride=(2, 2))
            , nn.BatchNorm2d(96))


        self.fc1 = nn.Sequential(
            nn.Linear(960, int(self.user_number*2/3))
            , nn.ReLU())

        # self.fc2 = nn.Sequential(
        #     nn.Linear(192, 192)
        #     , nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(int(self.user_number*2/3), self.user_number)
            )

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        # input x : 23 x 59049 x 1
        # expected conv1d input : minibatch_size x num_channel x width
        batch_size = x.size(0)
        # x = x.view(x.shape[0], 1, 8, -1)
        # x : batch_size x 1 x 900

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(batch_size, -1)  # flatten the tensor
        embedded = self.fc1(out)
        out = self.fc2(embedded)
        out = F.log_softmax(out, 1)
        return out, embedded


class Gait_Verifi_CNN_120_4_OU(torch.nn.Module):

    def __init__(self, user_number):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Gait_Verifi_CNN_120_4_OU, self).__init__()
        self.user_number = user_number
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(6, 1), stride=(1, 1), padding=(0, 0))
            , nn.ReLU()
            , nn.BatchNorm2d(64))

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=(6, 3), stride=(1, 1), padding=(2, 1))
            , nn.BatchNorm2d(96)
            , nn.MaxPool2d((3, 2), stride=(3, 2))
            , nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=(6, 2), stride=(1, 1), padding=(0, 0))
            , nn.BatchNorm2d(128)
            , nn.MaxPool2d((3, 1), stride=(3, 1))
            , nn.ReLU())

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 160, kernel_size=(3, 1), stride=(1, 1), padding=(0, 0))
            , nn.BatchNorm2d(160)
            # , nn.MaxPool2d((1, 3), stride=(1, 3))
            , nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(1440, 2048)
            , nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(2048, self.user_number)
            )

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        # input x : 23 x 59049 x 1
        # expected conv1d input : minibatch_size x num_channel x width
        batch_size = x.size(0)
        x = x.view(x.shape[0], 1, -1, 4)
        # x : batch_size x 1 x 900

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(batch_size, -1)  # flatten the tensor
        embedded = self.fc1(out)
        out = self.fc2(embedded)
        out = F.log_softmax(out, 1)
        return out, embedded


