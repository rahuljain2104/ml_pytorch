# objective is to make end 2 end system execute without fail
# import libraries
from keras.preprocessing import sequence
import pandas as pd
import pickle
import numpy as np
from normalization import normalize_documents
from utils import build_dataset
from numpy import array
from numpy import argmax
from keras.utils import to_categorical
import torch
from torch import nn
from torch.nn import LSTM

# defining parameters
vocabulary_size = 1500  # len(vocabulary)
max_doc_len = 50
epoch = 200
seq_len = max_doc_len
input_size = vocabulary_size
batch_size = 10
hidden_size = 150
num_layers = 2

dataset = pd.read_csv('kaggle_text_classification.csv', encoding="ISO-8859-1")

# dataset = pd.DataFrame(dataset)
# load data # get the data - csv
# df = pd.read_csv("movie_reviews_small.csv", encoding="ISO-8859-1")

# prepare training and test data
# preparing test labels
y = np.array(dataset["class"])
no_of_records = len(y)
no_of_classes = len(set(y))

# preparing test features
X = np.array(dataset["text"])
normalized_documents = normalize_documents(X)

def doc_to_onehot(documents):
    vocabulary = set(' '.join(list(documents)).split(' '))
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)

    # with open('filename.pickle', 'wb') as handle:
    #     pickle.dump([dictionary, reverse_dictionary, data],handle)
    X_train = []
    for doc in list(documents):
       tmp = []
       for word in doc.split(' '):
           try:
               tmp.append(dictionary[word])
           except:
               pass
       X_train.append(tmp)

    X_train = sequence.pad_sequences(X_train, maxlen=max_doc_len)

    data = array(X_train)
    # one hot encode
    encoded = to_categorical(data, num_classes=vocabulary_size)
    # invert encoding
    # inverted = argmax(encoded[0])
    return encoded

# x = np.zeros([batch_size, seq_len, input_size])
# y = np.zeros([batch_size])
x = doc_to_onehot(normalized_documents)

# generate model simple RNN or LSTM
class RNN(nn.Module):
    # feed data to the network
    def __init__(self, input_size, num_layers, hidden_size, no_of_classes):
        super(RNN, self).__init__()
        self.no_of_classes = no_of_classes
        self.rnn = LSTM(input_size,hidden_size=hidden_size,num_layers=num_layers, batch_first=True, bias=True, bidirectional=False)
        self.fc1_in = nn.Linear(in_features=hidden_size*seq_len, out_features=hidden_size, bias=True)
        self.fc_out = nn.Linear(in_features=hidden_size, out_features=no_of_classes, bias=True)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size)
        C0 = torch.zeros(num_layers, x.size(0), hidden_size)
        rnn_output, _ = self.rnn(x, (h0, C0))
        rnn_output = torch.reshape(rnn_output[:,:,:], (x.size(0),-1,))
        output = self.fc1_in(rnn_output)
        output = self.fc_out(output)
        return output

model = RNN(input_size, num_layers, hidden_size, no_of_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001)

loaded_model = RNN(input_size, num_layers, hidden_size, no_of_classes)
loaded_model.load_state_dict(torch.load('kaggle_text_classification.ckpt'))
print(loaded_model(torch.tensor(x).float()))

# print(model(torch.tensor(x[0:20]).float()))
#
# for t in range(epoch):
#     for i in range(int(no_of_records/batch_size)):
#         x_batch = torch.tensor(x[i*batch_size:(i+1)*batch_size]).float()
#         y_batch = torch.tensor(y[i*batch_size:(i+1)*batch_size]).long()
#         y_pred = model(x_batch)
#
#         loss = criterion(y_pred, y_batch)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # print(loss.item())
#         if (i + 1) % 4 == 0:
#             print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
#                   .format(t + 1, epoch, i + 1, int(no_of_records/batch_size), loss.item()))
#
# torch.save(model.state_dict(), 'kaggle_text_classification.ckpt')
#


# convert words in document to one hot representation
# prepare py-torch LSTM network
# feed data and get output - it should be binary
