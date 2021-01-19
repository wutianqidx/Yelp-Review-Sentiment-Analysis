import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
#import torch.distributed as dist

import time
import os
import sys
import io

from lstm_model import LSTM_model
#from rnn_model import RNN_model

SEED = 7
np.random.seed(SEED)
torch.manual_seed(SEED)

vocab_size = 8000
sequence_len = 150

x_train = np.load('../preprocessed_data/x_train.npy')
y_train = np.load('../preprocessed_data/y_train.npy')

#x_train = x_train[:10000]
#y_train = y_train[:10000]
x_test = np.load('../preprocessed_data/x_test.npy')
y_test = np.load('../preprocessed_data/y_test.npy')

vocab_size += 1

model = LSTM_model(vocab_size,800)
#model = LSTM_model(vocab_size,500)
#model = RNN_model(vocab_size,200)
model.cuda()

LR = 0.001
optimizer = optim.Adam(model.parameters(), lr=LR)

batch_size = 200
no_of_epochs = 20
L_Y_train = len(y_train)
L_Y_test = len(y_test)

model.train()

train_loss = []
train_accu = []
test_accu = [0]

for epoch in range(no_of_epochs):

    # training
    model.train()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0

    time1 = time.time()
    
    I_permutation = np.random.permutation(L_Y_train)

    for i in range(0, L_Y_train, batch_size):

        x_input = [x_train[j] for j in I_permutation[i:i+batch_size]]
        y_input = np.asarray([y_train[j] for j in I_permutation[i:i+batch_size]],dtype=np.int)
        data = Variable(torch.LongTensor(x_input)).cuda()
        target = Variable(torch.FloatTensor(y_input)).cuda()

        optimizer.zero_grad()
        loss, pred = model(data,target)
        loss.backward()
  
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if 'step' in state.keys():
                    if(state['step']>=1024):
                        state['step'] = 1000

        norm = nn.utils.clip_grad_norm_(model.parameters(),2.0)

        optimizer.step()   # update weights
        
        prediction = pred >= 0.0
        truth = target >= 0.5
        acc = prediction.eq(truth).sum().cpu().data.numpy()
        epoch_acc += acc
        epoch_loss += loss.data.item()
        epoch_counter += batch_size
    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter/batch_size)

    train_loss.append(epoch_loss)
    train_accu.append(epoch_acc)

    print(epoch, "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss, "%.4f" % float(time.time()-time1))
    
    model.eval()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0

    time1 = time.time()
    
    I_permutation = np.random.permutation(L_Y_test)

    for i in range(0, L_Y_test, batch_size):

        x_input = [x_test[j] for j in I_permutation[i:i+batch_size]]
        y_input = np.asarray([y_test[j] for j in I_permutation[i:i+batch_size]],dtype=np.int)
        target = Variable(torch.FloatTensor(y_input)).cuda()
        data = Variable(torch.LongTensor(x_input)).cuda()
        with torch.no_grad():
            loss, pred = model(data,target)
        
        prediction = pred >= 0.0
        truth = target >= 0.5
        acc = prediction.eq(truth).sum().cpu().data.numpy()
        epoch_acc += acc
        epoch_loss += loss.data.item()
        epoch_counter += batch_size

    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter/batch_size)

    test_accu.append(epoch_acc)

    time2 = time.time()
    time_elapsed = time2 - time1

    print("test", "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss, "%.4f" % float(time.time()-time1))
    if test_accu[-1] < test_accu[-2]:
        break
    
    torch.save(model,'LSTM.model')
