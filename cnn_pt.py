import os
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

#if it comes as a 784 long tensor or something x.reshape(-1,1,28,28)
#downloading data from pytorch mnist dataset

def cnn_main(X_train_2d, X_test_2d, X_valid_2d, y_train, y_test, y_valid, INPUT_DIM):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    if device=='xpu':
        device='cpu'
    print(f"Using {device} device")

    X_train_2d = torch.from_numpy(X_train_2d.astype(np.float32))
    y_train = torch.from_numpy(y_train)
    X_test_2d = torch.from_numpy(X_test_2d.astype(np.float32))
    y_test = torch.from_numpy(y_test)
    X_valid_2d = torch.from_numpy(X_valid_2d.astype(np.float32))
    y_valid = torch.from_numpy(y_valid)

    train_ds=TensorDataset(X_train_2d, y_train)
    test_ds=TensorDataset(X_test_2d, y_test)
    valid_ds=TensorDataset(X_valid_2d, y_valid)
    torch.manual_seed(1)
    batch_size = 100
    train_dl=DataLoader(train_ds, batch_size, shuffle=True)
    test_dl=DataLoader(test_ds, batch_size, shuffle=True)
    valid_dl=DataLoader(valid_ds, batch_size, shuffle=True)

    print("MNIST train -  rows:",X_train_2d.shape[0]," columns:", X_train_2d.shape[1:4])
    print("MNIST valid -  rows:",X_valid_2d.shape[0]," columns:", X_valid_2d.shape[1:4])
    print("MNIST test -  rows:",X_test_2d.shape[0]," columns:", X_test_2d.shape[1:4])

    #i want a cnn with 2 convolution layers, 2 max pooling layers, 2 dropout layers, 2 fully connected layers
    #defining model
    model=nn.Sequential()
    model.add_module(
        'conv1',
        nn.Conv2d(
            1, 32, 
            kernel_size=5, padding=2
        )
    )
    model.add_module('relu1', nn.ReLU())
    model.add_module('bn1',nn.BatchNorm2d(32))
    model.add_module('pool1', nn.MaxPool2d(kernel_size=2))
    model.add_module(
        'conv2',
        nn.Conv2d(
            32,64,
            kernel_size=5, padding=2
        )
    )
    model.add_module('relu2', nn.ReLU())
    model.add_module('pool2', nn.MaxPool2d(kernel_size=2))
    model.add_module('bn2', nn.BatchNorm2d(64))

    #start of dense layers
    model.add_module('flatten', nn.Flatten())
    #to calculate the size of the input of the first fully connected layer
    dummy_input=torch.zeros(1,1,INPUT_DIM,INPUT_DIM)
    dummy_output=model(dummy_input)
    flat_size=dummy_output.numel()
    if (INPUT_DIM>20):
        intermed_dim=512
    else:
        intermed_dim=1024
    model.add_module('fc1', nn.Linear(flat_size, intermed_dim))
    model.add_module('relu3', nn.ReLU())
    model.add_module('drop1', nn.Dropout(p=0.5))
    model.add_module('fc2', nn.Linear(intermed_dim, 10))
    #there is already softmax inside crossentropy
    model.to(device)
    loss_fn=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001)


    #TRAINING
    epochs=20
    start_time = time.perf_counter()
    loss_train=np.zeros(epochs)
    loss_valid=np.zeros(epochs)
    accuracy_train=np.zeros(epochs)
    accuracy_valid=np.zeros(epochs)
    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_dl:
            x_batch=x_batch.to(device)
            y_batch=y_batch.to(device)
            y_batch=y_batch.flatten()
            pred=model(x_batch)
            loss=loss_fn(pred, y_batch )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #mi dà la loss già normalizzata sulla dimensione della batch
            loss_train[epoch] += loss.item()*y_batch.size(0) 
            is_accurate= (torch.argmax(pred, dim=1)==y_batch).float()
            accuracy_train[epoch]+=is_accurate.sum()
        #len(train_dl) restituisce il numero dei batches 
        loss_train[epoch] /= len(train_dl.dataset)
        accuracy_train[epoch] /= len(train_dl.dataset)
        print(f"EPOCH: {epoch+1}/{epochs}")
        print("-----TRAINING------")
        print(f"Training loss: {loss_train[epoch]:.4f}\nTraining accuracy: {accuracy_train[epoch]:.4f}")
        model.eval()
        
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                x_batch=x_batch.to(device)
                y_batch=y_batch.to(device)
                y_batch=y_batch.flatten()
                pred2=model(x_batch)
                loss=loss_fn(pred2, y_batch)
                loss_valid[epoch]+=loss.item()*y_batch.size(0)
                is_correct=(torch.argmax(pred2, dim=1)==y_batch).float()
                accuracy_valid[epoch] += is_correct.sum()
            loss_valid[epoch] /= len(valid_dl.dataset)
            accuracy_valid[epoch] /=len(valid_dl.dataset)
        print("-----VALIDATION------")
        print(f"Valid loss: {loss_valid[epoch]:.4f}\nValid accuracy: {accuracy_valid[epoch]:.4f}")
        print("--------------------------------------------------")
    fig, ax= plt.subplots(1,2)
    end_time = time.perf_counter()
    print(f"Training + validation time : {end_time - start_time:.6f} seconds")

    ax[0].plot(loss_train, '-o', label='Train loss')
    ax[0].plot(loss_valid, '--<', label='Valid loss')
    ax[1].plot(accuracy_train, '-o', label='Train accuracy')
    ax[1].plot(accuracy_valid, '--<', label='Valid accuracy')
    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlabel('Epoch')
    ax[1].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[1].set_ylabel('Loss')
    
    fig.suptitle("Training/Validation loss and accuracy", fontsize=16)
    plt.show()

    #testing
    test_time=time.perf_counter()
    model.eval()
    accuracy_test=0
    with torch.no_grad():
        for x_batch, y_batch in test_dl:
            x_batch, y_batch=x_batch.to(device), y_batch.to(device)
            y_batch=y_batch.flatten()
            pred=model(x_batch)
            is_correct=(torch.argmax(pred, dim=1)==y_batch).float()
            accuracy_test += is_correct.sum()
        accuracy_test /= len(test_dl.dataset)
    end_test_time=time.perf_counter()
    print(f"Testing accuracy: {accuracy_test:.4f}")
    print(f"Testing time : {end_test_time - test_time:.6f} seconds")
    


