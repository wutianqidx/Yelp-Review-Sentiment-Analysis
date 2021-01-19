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

class att_LSTM_model(nn.Module):
    def __init__(self,vocab_size,n_hidden):
        super(att_LSTM_model, self).__init__()

        self.embedding = nn.Embedding(vocab_size,n_hidden)#,padding_idx=0)

        #self.lstm = nn.LSTM(n_hidden,n_hidden)
        self.lstm = nn.LSTM(n_hidden,n_hidden,num_layers=2,dropout=0.4)
        self.fc_output = nn.Linear(n_hidden*2, 1)

        #self.loss = nn.CrossEntropyLoss()
        self.loss = nn.BCEWithLogitsLoss()

     # lstm_output : [batch_size, n_step, n_hidden * num_directions(=1)], F matrix
    def attention_net(self, lstm_output, final_state,n_hidden):
        hidden = final_state.view(-1, n_hidden * 1, 1*2)   # hidden : [batch_size, n_hidden * num_directions(=1), 2(=n_layer)]
        attn_weights = torch.bmm(lstm_output, hidden)
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights)
        return context, soft_attn_weights.cpu().data.numpy() # context : [batch_size, n_hidden * num_directions(=1)*2]

    def forward(self, X, t, train=True):

        embed = self.embedding(X) # batch_size, time_steps, features
        no_of_timesteps = embed.shape[1]
        n_hidden = embed.shape[2]
        input = embed.permute(1, 0, 2) # input : [len_seq, batch_size, embedding_dim]
        hidden_state = Variable(torch.zeros(2*1, len(X), n_hidden)).cuda() # [num_layers(=2) * num_directions(=1), batch_size, n_hidden]
        cell_state = Variable(torch.zeros(2*1, len(X), n_hidden)).cuda() # [num_layers(=2) * num_directions(=1), batch_size, n_hidden]
        # final_hidden_state, final_cell_state : [num_layers(=2) * num_directions(=1), batch_size, n_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1, 0, 2) # output : [batch_size, len_seq, n_hidden]
        h, attention = self.attention_net(output, final_hidden_state,n_hidden)
        h = h.view(h.size(0),-1)
        h = self.fc_output(h)
        return self.loss(h[:,0],t), h[:,0]#F.softmax(h, dim=1)

