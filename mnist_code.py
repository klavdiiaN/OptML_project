import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# defining MLP class

class MLP(nn.Module):
    
    def __init__(self, hidden_size_1=100, hidden_size_2=10):
        super(MLP, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, hidden_size_1),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size_1, hidden_size_2))
    
    # forward pass
    def forward(self, x):
        return self.layers(x)

mlp = MLP()
print(mlp)

# defining accuracy function

def accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

# MLP implementation function

def mlp_nn(x_train, y_train, x_test, y_test, model, optimizer, criterion, num_epoch):
    loss_all_train, loss_all_test = [], []
    epochs_all = torch.arange(1, num_epoch+num_epoch/10, num_epoch/10)
    epochs_all[-1] = num_epoch - 1
            
    for epoch in range(num_epoch):
        y = model(x_train)
        loss_train = criterion(y, y_train)
        
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        if epoch % 100 == 0:
            loss_train = loss_train.detach().numpy()
            loss_all_train.append(loss_train)

            y_test_obt = model(x_test)
            loss_test = criterion(y_test_obt, y_test)
            loss_test = loss_test.detach().numpy()
            loss_all_test.append(loss_test)
            acc = accuracy(y_test_obt, y_test)
            
            print ('Epoch [%d/%d], Train Loss: %.4f, Test Loss: %.4f, Accuracy: %.4f' 
            %(epoch+1, num_epoch, loss_train, loss_test, acc))

        if epoch == num_epoch - 1:
            loss_train = loss_train.detach().numpy()
            loss_all_train.append(loss_train)

            y_test_obt = model(x_test)
            loss_test = criterion(y_test_obt, y_test)
            loss_test = loss_test.detach().numpy()
            loss_all_test.append(loss_test)
            acc = accuracy(y_test_obt, y_test)
            
            print('Final, Train Loss: %.4f, Test Loss: %.4f, Accuracy: %.4f' %(loss_train, loss_test, acc))

            # plotting train and test loss

            fig, axs = plt.subplots(1, 2, figsize=(10, 8))
            axs[0].plot(epochs_all, loss_all_train, linewidth=2.5, color='blue')
            axs[0].set_ylabel('Train Loss', fontsize=12)
            axs[0].set_xlabel('Epoch', fontsize=12)

            axs[1].plot(epochs_all, loss_all_test, linewidth=2.5, color='orange')
            axs[1].set_ylabel('Test Loss', fontsize=12)
            axs[1].set_xlabel('Epoch', fontsize=12)
            fig.suptitle('MLP Convergence', fontsize=15)
            plt.show()

    return loss_all_train, loss_all_test

model = MLP()
criterion = nn.CrossEntropyLoss() # good loss function for classification tasks
num_epoch = 1000
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), weight_decay = 1e-5, lr=learning_rate) # weight_decay is an L2 regularisation to avoid overfitting
