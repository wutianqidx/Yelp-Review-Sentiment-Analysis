import numpy as np
import io

np.random.seed(8)
vocab_size = 8000
sequence_len = 150

x_train = []
y_train = []

def pad_sequence(x, seq_len = 150):
    if len(x) > seq_len:
        start_index = np.random.randint(len(x)-seq_len+1)
        #x_input[j,:] = x[start_index:(start_index+sequence_length)]
        #return x[:seq_len]
        return x[start_index:(start_index+seq_len)]
    else:
        return x + [0]*(seq_len-len(x))

#count = 0

with io.open('preprocessed_data/yelp_train.txt','r',encoding='utf-8') as f:
    for line in f:
        label, review = line.strip().split(',')
        if review != '':
            y_train.append(int(label[1])-1)
            ids = list(map(lambda x: int(x), pad_sequence(review.split())))
            x_train.append(ids)
        #count +=1
        #if count==10000:
        #    break
    f.close()

x_train = np.array(x_train)
x_train[x_train > vocab_size] = 0
y_train = np.array(y_train)

x_test, y_test = [], []
with io.open('preprocessed_data/yelp_test.txt','r',encoding='utf-8') as f:
    for line in f:
        label, review = line.strip().split(',')
        if review != '':
            y_test.append(int(label[1])-1)
            ids = list(map(lambda x: int(x), pad_sequence(review.split(),450)))
            x_test.append(ids)
    f.close()
x_test = np.array(x_test)
x_test[x_test > vocab_size] = 0
y_test = np.array(y_test)

np.save('preprocessed_data/x_train.npy',x_train)
np.save('preprocessed_data/y_train.npy',y_train)
np.save('preprocessed_data/x_test.npy',x_test)
np.save('preprocessed_data/y_test.npy',y_test)
