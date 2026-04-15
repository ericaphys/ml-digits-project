import torchvision
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
import torch.nn as nn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2= nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        #x = self.layer1(x)
        #x= nn.ReLU()(x)
        x = torch.relu(self.layer1(x))
        x=self.layer2(x)
        return x
    


def pytorch_main(X_train, y_train, X_test, y_test, hidden_size, eta):
    start_time = time.perf_counter()
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    if device=='xpu':
        device='cpu'
    print(f"Using {device} device")
    #device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    #print(f"Using {device} device")
    
    '''
    X, y=fetch_openml('mnist_784', version=1, return_X_y=True)
    X = X.values
    y = y.astype(int).values

    X = ((X / 255.) - .5) * 2

    #X, y = X.to(device), y.to(device)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1./3, random_state=123, stratify=y)

    # optional to free up some memory by deleting non-used arrays:
    del X, y
    '''
    #pytorching
    X_train = torch.from_numpy(X_train.astype(np.float32))
    y_train = torch.from_numpy(y_train)
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_test = torch.from_numpy(y_test)
    
    train_ds=TensorDataset(X_train, y_train)
    test_ds=TensorDataset(X_test, y_test)
    torch.manual_seed(1)
    batch_size = 100
    train_dl=DataLoader(train_ds, batch_size, shuffle=True)
    test_dl=DataLoader(test_ds, batch_size, shuffle=True)

    input_size=X_train.shape[1]
    #hidden_size=50
    output_size=10

    model=Model(input_size, hidden_size, output_size)
    model.to(device)

    #eta=0.001

    #cross-entropy (logloss) instead of loss and adam optimizer instead of stochastic gradient descent 
    loss_fn=nn.CrossEntropyLoss()   

    optimizer=torch.optim.Adam(model.parameters(), lr=eta)


    #---------- training ------------
    epochs=100
    losses=np.zeros(epochs, dtype=np.float32)
    accuracy=np.zeros(epochs, dtype=np.float32)
    for ep in range(epochs):
        for x_batch, y_batch in train_dl:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred=model(x_batch)
            loss= loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses[ep] += loss.item()*y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1)==y_batch)
            accuracy[ep] += is_correct.sum()

        losses[ep] /= len(train_dl.dataset)
        accuracy[ep] /= len(train_dl.dataset)
        print(f"Epoch: {ep+1}/{epochs}\nAccuracy : {accuracy[ep]*100}%\n----------------------------\n")

    end_time = time.perf_counter()
    print(f"Training + loading data time : {end_time - start_time:.6f} seconds")
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(losses, lw=3)
    ax.set_title('Training loss', size=15)
    ax.set_xlabel('Epoch', size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(accuracy, lw=3)
    ax.set_title('Training accuracy', size=15)
    ax.set_xlabel('Epoch', size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()

    #plt.savefig('figures/12_09.pdf')
 
    plt.show()

    start_time=time.perf_counter()
    model.eval()  # Set to evaluation mode
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient calculation
        for images, labels in test_dl:
            images = images.view(images.size(0), -1)
            images, labels =images.to(device), labels.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            total += labels.size(0)
            correct += (predictions == labels.view(-1)).sum().item()



        is_correct=correct/total
        accuracy=is_correct*100
        
        print(f"Test accuracy: {accuracy:.4f}%")
    end_time=time.perf_counter()
    print(f"Testing time: {end_time - start_time:.6f} seconds")

    #-------- saving model ---------
    path='digits_emnist_mlp1.pt'
    torch.save(model.state_dict(), path)



    return 0

if __name__=='__main__':
    print("ciao")