import numpy as np
import os
import nltk
import itertools
import io
import re

if(not os.path.isdir('preprocessed_data')):
    os.mkdir('preprocessed_data')
file_path = 'yelp_review_polarity_csv/'
x_train = []
y_train = []
with io.open(file_path+'train.csv') as f:
    for line in f:
        l = line.strip().split(',')
        senti, text = l[0], re.sub('[^a-zA-Z \']', '', re.sub('\\\\n', ' ', ','.join(l[1:]))).lower()
        tokens = nltk.word_tokenize(text)
        x_train.append(tokens)
        y_train.append(senti)
    f.close()

x_test = []
y_test = []
with io.open(file_path+'test.csv') as f:
    for line in f:
        l = line.strip().split(',')
        senti, text = l[0], re.sub('[^a-zA-Z \']', '', re.sub('\\\\n', ' ', ','.join(l[1:]))).lower()
        tokens = nltk.word_tokenize(text)
        x_test.append(tokens)
        y_test.append(senti)
    f.close()

#len(x_train), len(x_test)

## number of tokens per review
#no_of_tokens = []
#for tokens in x_train:
#    no_of_tokens.append(len(tokens))
#no_of_tokens = np.asarray(no_of_tokens)
#print('Total: ', np.sum(no_of_tokens), ' Min: ', np.min(no_of_tokens), ' Max: ',
# np.max(no_of_tokens), ' Mean: ', np.mean(no_of_tokens), ' Std: ', np.std(no_of_tokens))


all_tokens = itertools.chain.from_iterable(x_train)
word_to_id = {token: idx for idx, token in enumerate(set(all_tokens))}

all_tokens = itertools.chain.from_iterable(x_train)
id_to_word = [token for idx, token in enumerate(set(all_tokens))]
id_to_word = np.asarray(id_to_word)

x_train_token_ids = [[word_to_id[token] for token in x] for x in x_train]
count = np.zeros(id_to_word.shape)
for x in x_train_token_ids:
    for token in x:
        count[token] += 1
indices = np.argsort(-count)
id_to_word = id_to_word[indices]
count = count[indices]

hist = np.histogram(count,bins=[1,10,100,1000,10000])
print(hist)
for i in range(10):
    print(id_to_word[i],count[i])

## recreate word_to_id based on sorted list
word_to_id = {token: idx for idx, token in enumerate(id_to_word)}

## assign -1 if token doesn't appear in our dictionary
## add +1 to all token ids, we went to reserve id=0 for an unknown token
x_train_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_train]
x_test_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in x_test]

## save dictionary
np.save('preprocessed_data/yelp_dictionary.npy',np.asarray(id_to_word))

## save training data to single text file
with io.open('preprocessed_data/yelp_train.txt','w',encoding='utf-8') as f:
    for i in range(len(x_train_token_ids)):
        f.write(y_train[i]+',')
        for token in x_train_token_ids[i]:
            f.write("%i " % token)
        f.write("\n")
    f.close()

## save test data to single text file
with io.open('preprocessed_data/yelp_test.txt','w',encoding='utf-8') as f:
    for i in range(len(x_test_token_ids)):
        f.write(y_test[i]+',')
        for token in x_test_token_ids[i]:
            f.write("%i " % token)
        f.write("\n")
    f.close()
