import pandas as pd
from sklearn import datasets
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
import numpy as np
#import scipy as sc
import emnist
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
import sys

import multiclass_perceptron_numpy_opt as perceptron
import multiclass_adaline_sgd as adaline
import multiclass_log_reg as log_reg
import supp_v_m as svm
from cnn_pt import cnn_main
from mlp import mlp_main
from mlp_pt import pytorch_main
import kagglehub
import mlp

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

def main():
    
    #-----------------------------------
    #--------- INPUT DATASETS ----------
    #-----------------------------------
    print("Select dataset: ")
    print("1- Scikit learn digits dataset")
    print("2- EMNIST digits dataset")
    print("3- MNIST digits dataset ")
    number=int(input("Your choice: "))

    if(number==1):    
        #------------scikit-learn------------------
        print("You have selected the Scikit learn digits dataset")
        digits = datasets.load_digits()
        X_train, X_test, y_train1, y_test1 = train_test_split(
        digits.data, digits.target, test_size=0.33, random_state=42)
        print(f"Train dataset dimension: {X_train.shape[0]}")
        #SCALING
        scale=np.float32(np.max(X_train))
        X_train =X_train/scale
        X_test=X_test/scale

        #STANDARDIZATION

        #feature by feature std
        X_train=(X_train-np.mean(X_train))/np.std(X_train)
        X_test=(X_test-np.mean(X_test))/np.std(X_test)
        X_train, X_valid, y_train1, y_valid = train_test_split(X_train, y_train1, test_size=1./11, random_state=123, stratify=y_train1)

        y_train=y_train1
        y_test=y_test1
        train_data=(np.column_stack((y_train1, X_train)))
        test_data=(np.column_stack((y_test1, X_test)))
        X_test_2d=X_test.reshape(-1,1,8,8)
        X_train_2d=X_train.reshape(-1,1,8,8)
        X_valid_2d=X_valid.reshape(-1, 1, 8, 8)
        hidden=32
        eta=1
        input_dim=8
    
    elif(number==2):
        print("You have selected the EMNIST digits dataset")
        #---------emnist data csv--------------------
        #train_data_csv=pd.read_csv(r'C:\Users\novel\.cache\kagglehub\datasets\crawford\emnist\versions\3\emnist-digits-train.csv')
        #train_data=train_data_csv.to_numpy()
    
        #test_data_csv=pd.read_csv(r'C:\Users\novel\.cache\kagglehub\datasets\crawford\emnist\versions\3\emnist-digits-test.csv')
        #test_data=test_data_csv.to_numpy()
    
        
    
        #-------------emnist lib------------
        images, labels = emnist.extract_training_samples('digits')
        X_train = images.reshape(images.shape[0], 784)
        y_train = labels.reshape(-1, 1)

        images_test, labels_test = emnist.extract_test_samples('digits')
        X_test = images_test.reshape(images_test.shape[0], 784)
        y_test = labels_test.reshape(-1, 1)
        print(f"Train dataset dimension: {X_train.shape[0]}")
        #SCALING
        scale=np.float32(np.max(X_train))
        X_train =X_train/scale
        X_test=X_test/scale
    
        #STANDARDIZATION
        X_train=(X_train-np.mean(X_train))/np.std(X_train)
        X_test=(X_test-np.mean(X_test))/np.std(X_test)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=1./11, random_state=123, stratify=y_train)

        y_train1=y_train.ravel()
        y_test1=y_test.ravel()

        train_data = np.concatenate((y_train, X_train), axis=1)
        test_data = np.concatenate((y_test, X_test), axis=1)
        X_test_2d=X_test.reshape(-1,1,28,28)
        X_train_2d=X_train.reshape(-1,1,28,28)
        X_valid_2d=X_valid.reshape(-1,1,28,28)
        hidden=128
        eta=0.1
        input_dim=28
    else:
        print("You have selected the MNIST digits dataset")
        X, y=fetch_openml('mnist_784', version=1, return_X_y=True)
        X = X.values
        y = y.astype(int).values

        X = ((X / 255.) - .5) * 2

        #X, y = X.to(device), y.to(device)
        X_train, X_test, y_train, y_test1 = train_test_split(X, y, test_size=1./3, random_state=123, stratify=y)
        X_train, X_valid, y_train1, y_valid = train_test_split(X_train, y_train, test_size=1./11, random_state=123, stratify=y_train)
        y_train = y_train1.reshape(-1, 1)
        y_test = y_test1.reshape(-1, 1)

        train_data = np.concatenate((y_train, X_train), axis=1)
        test_data = np.concatenate((y_test, X_test), axis=1)
        print(f"Train dataset dimension: {X_train.shape[0]}")

        X_test_2d=X_test.reshape(-1,1,28,28)
        X_train_2d=X_train.reshape(-1,1,28,28)
        X_valid_2d=X_valid.reshape(-1,1,28,28)
        # optional to free up some memory by deleting unused arrays:
        del X, y
        hidden=64
        eta=0.1
        input_dim=28

    



    #----------------------------------------
    #------------- MODEL CHOICE -------------
    #----------------------------------------

    print("Select model: ")
    #creati da me 
    print("1- Single layer Perceptron")
    print("2- Single layer Adaline")
    print("3- Logistic regression")
    #non creato da me
    print("4- Support vector machine (linear or radial kernels available)")
    print("5- Pytorch MLP (1 hidden layer)")
    print("6- MLP homemade (1 hidden layer)")
    print("7- CNN")
    model=int(input("Your choice: "))
    
    if(model<4):
        if(model==1):
            ppn = perceptron.Perceptron(eta=0.1, n_iter=100)
        elif(model==2):
            ppn = adaline.Perceptron(eta=0.0001, n_iter=400)
        else:
            ppn = log_reg.Perceptron(eta=0.01, n_iter=100)
        start_time = time.perf_counter()
        ppn.fit(train_data)
        ppn.test(test_data)
        end_time = time.perf_counter()
        print(f"Training + testing time : {end_time - start_time:.6f} seconds")
    
    elif(model==4):
        #SUPPORT VECTOR MACHINE
        ker=int(input("Enter kernel (0- linear, 1- rbf)"))
        supp=svm.Support_vector_machine()
        if(ker==0):
            start_time = time.perf_counter()
            supp.fit_predict(X_train, X_test, y_train, y_test, 'linear', 1.0)
            end_time = time.perf_counter()
            print(f"Training + testing time : {end_time - start_time:.6f} seconds")
        else:
            start_time = time.perf_counter()
            supp.fit_predict(X_train, X_test, y_train, y_test, 'rbf', 1.0)
            end_time = time.perf_counter()
            print(f"Training + testing time : {end_time - start_time:.6f} seconds")
    elif(model==5):
        
        pytorch_main(X_train, y_train1, X_test, y_test1, hidden, eta=0.001 )
        sys.exit()

    elif(model==6):

        mlp_main(X_train, y_train, X_valid, y_valid, X_test, y_test, hidden, eta=1)
        sys.exit()
    elif(model==7):
        cnn_main(X_train_2d, X_test_2d, X_valid_2d, y_train, y_test, y_valid, input_dim)

    
if __name__=="__main__":
    main()