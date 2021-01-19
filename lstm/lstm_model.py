import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

import time
import os
import sys
import io

class LSTM_model(nn.Module):
    def __init__(self,vocab_size,n_hidden):
        super(LSTM_model, self).__init__()

        self.embedding = nn.Embedding(vocab_size,n_hidden)#,padding_idx=0)

        #self.lstm = nn.LSTM(n_hidden,n_hidden)
        self.lstm = nn.LSTM(n_hidden,n_hidden,num_layers=2,dropout=0.4)
        self.fc_output = nn.Linear(n_hidden, 1)

        #self.loss = nn.CrossEntropyLoss()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, X, t, train=True):

        embed = self.embedding(X) # batch_size, time_steps, features
        no_of_timesteps = embed.shape[1]
        n_hidden = embed.shape[2]
        input = embed.permute(1, 0, 2) # input : [len_seq, batch_size, embedding_dim]
        hidden_state = Variable(torch.zeros(2*1, len(X), n_hidden)).cuda() # [num_layers(=2) * num_directions(=1), batch_size, n_hidden]
        cell_state = Variable(torch.zeros(2*1, len(X), n_hidden)).cuda() # [num_layers(=2) * num_directions(=1), batch_size, n_hidden]
        # final_hidden_state, final_cell_state : [num_layers(=2) * num_directions(=1), batch_size, n_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        #output = output.permute(1, 2, 0) # output : [batch_size, n_hidden, len_seq]
        h = output[-1]
        #pool = nn.MaxPool1d(no_of_timesteps)
        #h = pool(output)
        h = h.view(h.size(0),-1)
        h = self.fc_output(h)
        return self.loss(h[:,0],t), h[:,0]#F.softmax(h, dim=1)

