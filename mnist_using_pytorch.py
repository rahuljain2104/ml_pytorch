# get dataset - using pandas
import pandas as pd
import torch as th
from sklearn.model_selection import train_test_split
import numpy as np
from torch import nn


train_data = pd.read_csv('dataset/train.csv')

label = np.array(train_data['label'])
features = np.array(train_data.drop(['label'], axis=1))

# convert into numpy array
# train_data = np.array(train_data)
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.3)

num_input = 784
num_output = 10
hidden_layers = [400, 300]
epoch = 10
batch_size = 100
learning_rate = 0.05

# # Define model
class DNN_Model(nn.Module):
    
    def __init__(self, num_input, hidden_layers, num_output):
        super(DNN_Model, self).__init__()
        self.input = nn.Linear(in_features=num_input, out_features=hidden_layers[0])
        self.relu = nn.ReLU()
        self.input1 = nn.Linear(in_features=hidden_layers[0], out_features=hidden_layers[1])
        self.output = nn.Linear(in_features=hidden_layers[1], out_features=num_output)

    def forward(self, x):
        # print(x)
        out = self.input(x)
        out = self.relu(out)
        out = self.input1(out)
        out = self.relu(out)
        out = self.output(out)
        return out


model = DNN_Model(num_input, hidden_layers, num_output)
criterion = nn.CrossEntropyLoss()
optimizer = th.optim.Adagrad(model.parameters(), lr = learning_rate)
#
# for i in range(epoch):
#     for j in range(int(len(x_train)/batch_size)):
#         x_batch = th.Tensor(x_train[j*batch_size:(j+1)*batch_size])
#         y_batch = th.Tensor(y_train[j*batch_size:(j+1)*batch_size]).long()
#         # if len(x_train[j*batch_size:(j+1)*batch_size]) == 0:
#         #     continue
#         y_predicted = model(x_batch)
#         loss = criterion(y_predicted, y_batch)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if (j + 1) % 100 == 0:
#             print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
#                   .format(i + 1, epoch, j + 1, int(len(x_train)/batch_size), loss.item()))
#
#
# th.save(model.state_dict(), 'model_dnn_state.ckpt')
# th.save(model, 'model_dnn.ckpt')

# testing
model.load_state_dict(th.load('model_dnn_state.ckpt'))
y_predicted = model(th.Tensor(x_test))
print(y_predicted)
loss = criterion(y_predicted, th.Tensor(y_test).long())
print(loss)
# train DNN model
# define loss function
# calculate loss
# reset gradients

# save model
# testing on the model
# test_data = pd.read_csv('dataset/test.csv')

