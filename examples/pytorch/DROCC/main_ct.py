from __future__ import print_function
import numpy as np
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
from data_process_scripts.process_ct import CT_Dataset
from edgeml_pytorch.trainer.drocc_trainer import DROCCTrainer

class CT_LeNet(nn.Module):

    def __init__(self):
        super(CT_LeNet, self).__init__()

        self.rep_dim = 64
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(182, 32, 3, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 3, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(256, self.rep_dim, bias=False)
        self.fc2 = nn.Linear(self.rep_dim, int(self.rep_dim/2), bias=False)
        self.fc3 = nn.Linear(int(self.rep_dim/2), 1, bias=False)

    def forward(self, x):
        # print("0: ", x.size()) # []
        x = x.unsqueeze(3)
        # print("1: ", x.size()) # []
        x = self.conv1(x)
        # print("2: ", x.size()) # []
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        # print("3: ", x.size()) # []
        x = self.conv2(x)
        # print("4: ", x.size()) # []
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        # print("5: ", x.size()) # []
        x = self.conv3(x)
        # print("6: ", x.size()) # []
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        # print("7: ", x.size()) # []
        x = x.view(x.size(0), -1)
        # print("8: ", x.size()) # []
        # x = self.fc1(x)
        x = F.leaky_relu(self.fc1(x))
        # print("9: ", x.size()) # []
        x = F.leaky_relu(self.fc2(x))
        # print("10: ", x.size()) # []
        x = self.fc3(x)
        # print("11: ", x.size()) # []
        return x


# class LSTM_FC(nn.Module):
#     """
#     Single layer LSTM with a fully connected layer
#     on the last hidden state
#     """
#     def __init__(self,
#                  input_dim=3,
#                  num_classes=1, 
#                  num_hidden_nodes=8
#     ):

#         super(LSTM_FC, self).__init__()
#         self.input_dim = input_dim
#         self.num_classes = num_classes
#         self.num_hidden_nodes = num_hidden_nodes
#         self.encoder = nn.LSTM(input_size=self.input_dim, 
#             hidden_size=self.num_hidden_nodes,
#             num_layers=1, batch_first=True)
#         self.fc = nn.Linear(self.num_hidden_nodes,
#             self.num_classes)
#         activ = nn.ReLU(True)

#     def forward(self, input):
#         # print("input:", input.size())
#         features = self.encoder(input)[0][:,-1,:]
#         # print("features:", features.size())
#         # pdb.set_trace()
#         logits = self.fc(features)
#         # print("logits:", logits.size())
#         return logits

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return torch.from_numpy(self.data[idx]), (self.labels[idx]), torch.tensor([0])


def adjust_learning_rate(epoch, total_epochs, only_ce_epochs, learning_rate, optimizer):
        """Adjust learning rate during training.

        Parameters
        ----------
        epoch: Current training epoch.
        total_epochs: Total number of epochs for training.
        only_ce_epochs: Number of epochs for initial pretraining.
        learning_rate: Initial learning rate for training.
        """
        #We dont want to consider the only ce 
        #based epochs for the lr scheduler
        epoch = epoch - only_ce_epochs
        drocc_epochs = total_epochs - only_ce_epochs
        # lr = learning_rate
        if epoch <= drocc_epochs:
            lr = learning_rate * 0.01
        if epoch <= 0.80 * drocc_epochs:
            lr = learning_rate * 0.1
        if epoch <= 0.40 * drocc_epochs:
            lr = learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return optimizer


def main():
    # Loading CT Dataset
    dataset = CT_Dataset("data", args.normal_class)
    train_loader, test_loader = dataset.loaders(batch_size=args.batch_size)

    # Using LSTM model check dimension
    # model = LSTM_FC(input_dim=3, num_classes=1, num_hidden_nodes=args.hd).to(device)
    
    # Using CNN model
    model = CT_LeNet().to(device)
    model = nn.DataParallel(model)

    if args.optim == 1:
        optimizer = optim.SGD(model.parameters(),
                                  lr=args.lr,
                                  momentum=args.mom)
        print("using SGD")
    else:
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr)
        print("using Adam")
    
    trainer = DROCCTrainer(model, optimizer, args.lamda, args.radius, args.gamma, device)

    if args.eval == 0:
        # Training the model 
        trainer.train(train_loader, test_loader, args.lr, adjust_learning_rate, args.epochs,
            metric=args.metric, ascent_step_size=args.ascent_step_size, only_ce_epochs = 0)

        trainer.save(args.model_dir)
        
    else:
        if os.path.exists(os.path.join(args.model_dir, 'model.pt')):
            trainer.load(args.model_dir)
            print("Saved Model Loaded")
        else:
            print('Saved model not found. Cannot run evaluation.')
            exit()
        score = trainer.test(test_loader, 'AUC')
        print('Test AUC: {}'.format(score))

if __name__ == '__main__':
    torch.set_printoptions(precision=5)
    
    parser = argparse.ArgumentParser(description='PyTorch Simple Training')
    parser.add_argument('--normal_class', type=int, default=0, metavar='N',
                    help='CT normal class index')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size for training')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('-oce,', '--only_ce_epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train with only CE loss')
    parser.add_argument('--ascent_num_steps', type=int, default=100, metavar='N',
                        help='Number of gradient ascent steps')                        
    parser.add_argument('--hd', type=int, default=128, metavar='N',
                        help='Number of hidden nodes for LSTM model')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--ascent_step_size', type=float, default=0.001, metavar='LR',
                        help='step size of gradient ascent')                        
    parser.add_argument('--mom', type=float, default=0.0, metavar='M',
                        help='momentum')
    parser.add_argument('--model_dir', default='log',
                        help='path where to save checkpoint')
    parser.add_argument('--one_class_adv', type=int, default=1, metavar='N',
                        help='adv loss to be used or not, 1:use 0:not use(only CE)')
    parser.add_argument('--radius', type=float, default=0.2, metavar='N',
                        help='radius corresponding to the definition of set N_i(r)')
    parser.add_argument('--lamda', type=float, default=1, metavar='N',
                        help='Weight to the adversarial loss')
    parser.add_argument('--reg', type=float, default=0, metavar='N',
                        help='weight reg')
    parser.add_argument('--eval', type=int, default=0, metavar='N',
                        help='whether to load a saved model and evaluate (0/1)')
    parser.add_argument('--optim', type=int, default=0, metavar='N',
                        help='0 : Adam 1: SGD')
    parser.add_argument('--gamma', type=float, default=2.0, metavar='N',
                        help='r to gamma * r projection for the set N_i(r)')
    parser.add_argument('-d', '--data_path', type=str, default='.')
    parser.add_argument('--metric', type=str, default='AUC')
    args = parser. parse_args()

    # settings
    #Checkpoint store path
    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    main()