import numpy as np
import time
import emnist
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from numba import jit

#sigmoid activation function
#the same activation function is used for the hidden and output layers
#one can preferably use softmax for the output and relu for hidden
def activation(z):
    return 1./(1. +np.exp(-z))


def int_to_onehot(y, num_labels):
    target=np.zeros((y.shape[0], num_labels))
    y=y.astype(int)
    for i, val in enumerate(y):
        target[i,val]=1
    return target


def shuffle_time(X,y):
    r=np.random.permutation(len(y))
    return X[r], y[r]


def mini_batches_gen(X, y, size):
    X, y = shuffle_time(X,y)    #shuffle the dataset
    
    if(len(y)%size!=0):
        size=size-(len(y)%size!=0)
    for i in range(0,len(y),size):
        mini_batch_x=X[i:i+size, :]
        mini_batch_y=y[i:i+size]
        yield mini_batch_x, mini_batch_y


def compute_mse_and_acc(nnet, X, y, num_labels=10, minibatch_size=100):
    mse, correct_pred, num_examples = 0., 0, 0
    minibatch_gen = mini_batches_gen(X, y, minibatch_size)
        
    for i, (features, targets) in enumerate(minibatch_gen):

        _, probas = nnet.forward(features)
        predicted_labels = np.argmax(probas, axis=1)
        
        onehot_targets = int_to_onehot(targets, num_labels=num_labels)
        loss = np.mean((onehot_targets - probas)**2)
        correct_pred += (predicted_labels == targets).sum()
        
        num_examples += targets.shape[0]
        mse += loss

    mse = mse/(i+1)
    acc = correct_pred/num_examples
    return mse, acc



def training(model, X_train, y_train, X_valid, y_valid, epochs=100, mini_batch_size=100, eta=0.1):
    accs=[]
    accs_val=[]
    mses=[]  
    for i in range(epochs):
        #a_h, a_out= model.forward(X_train)
        mini_batches=mini_batches_gen(X_train,y_train,mini_batch_size)
        for X_train_mini, y_train_mini in mini_batches:
            #every iteration gives a different mini-batch for gradient descent
            a_hid, a_out =model.forward(X_train_mini)

            #computing the gradients 
            d_loss_d_w_out, d_loss_d_b_out, d_loss_d_w_h, d_loss_d_b_h = model.backward(X_train_mini, y_train_mini, a_hid, a_out)

            #updating weights and biases 
            model.w_out += -eta * d_loss_d_w_out
            model.w_hid += -eta * d_loss_d_w_h
            model.b_out += -eta * d_loss_d_b_out
            model.b_h += -eta * d_loss_d_b_h 
        #print(f"Accuracy score, epoch {i}: {accuracy_score(y_true, predictions)}")
        
        mse, acc=compute_mse_and_acc(model, X_train, y_train)
        mse, acc = mse.item(), acc.item()
        mse_val, acc_val=compute_mse_and_acc(model, X_valid, y_valid)
        mse_val, acc_val = mse_val.item(), acc_val.item()
        print(f"Epoch: {i+1:03d}/{epochs:03d}")
        print(f"Train MSE: {mse:.2f} | Train Acc: {acc*100:.2f}% | Valid Acc: {acc_val*100:.2f}%\n")
        accs.append(acc)
        accs_val.append(acc_val)
        mses.append(mse)

            
    return mse, accs, accs_val


#test
def testing(model, X_test, y_test):
    #a_hid, a_out =model.forward(X_test)
    k, acc=compute_mse_and_acc(model, X_test, y_test)
    print(f"Test set accuracy: {100*acc:.2f}%")
    return acc


class NeuralNetMLP:
    def __init__ (self, num_features, num_hidden, num_classes, random_seed=123):
        super().__init__()

        self.num_classes=num_classes

        #bias and weight are separate, one can insert a column in w_hid/out=np.insert(w_hid/out,0,bias,axis=1)
        #hidden layer weights initialization
        rng=np.random.RandomState(random_seed)
        self.w_hid=rng.normal(loc=0.0, scale=0.1, size=(num_hidden, num_features)).astype(np.float32)
        
        self.b_h=np.zeros(num_hidden).astype(np.float32)


        #output layer weights initialization
        
        self.w_out=rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden)).astype(np.float32)

        self.b_out=np.zeros(num_classes).astype(np.float32)

    
    def forward(self, X):
        #hidden layer
        dots_h=(np.dot(X, self.w_hid.T)+self.b_h).astype(np.float32)
        hid_output=activation(dots_h)

        #output layer
        dots_out=(np.dot(hid_output, self.w_out.T)+self.b_out).astype(np.float32)
        out_output=activation(dots_out)

        return hid_output, out_output

    #it is possible to swich to sgd/adam implementation
    #this method will work with mini-batches gradient descent for optimization
    #min of the MSE as loss function instead of cross-entropy
    def backward(self, X , y_label , hid_output, out_output):
        #one hot encoding nxp
        y_target=int_to_onehot(y_label, self.num_classes)
        #chain-rule for gradient descent
        d_loss_d_a_out= 2.*(out_output-y_target)/y_label.shape[0]
        d_a_out_d_z_out=out_output*(1.-out_output) #sigmoid derivative
        delta_out= d_loss_d_a_out*d_a_out_d_z_out
        d_z_out_d_w_out=hid_output
        d_loss_d_w_out=np.dot(delta_out.T, d_z_out_d_w_out)
        d_loss_d_b_out=np.sum(delta_out, axis=0)
        #hidden layer
        d_z_out_a_h=self.w_out
        d_loss_a_h=np.dot(delta_out, d_z_out_a_h)
        d_a_h_d_z_h=hid_output*(1.-hid_output)
        d_z_h_d_w_h=X
        d_loss_d_w_h=np.dot((d_loss_a_h*d_a_h_d_z_h).T,d_z_h_d_w_h)
        d_loss_d_b_h = np.sum((d_loss_a_h * d_a_h_d_z_h), axis=0)

        return d_loss_d_w_out, d_loss_d_b_out, d_loss_d_w_h, d_loss_d_b_h



    

def mlp_main(X_train, y_train, X_valid, y_valid, X_test, y_test, num_hidden, eta):
    start_time = time.perf_counter()
    '''
    #-------------emnist lib------------
    images, labels = emnist.extract_training_samples('digits')
    X_train = images.reshape(images.shape[0], 784)
    y_train = labels.reshape(-1, 1)

    images_test, labels_test = emnist.extract_test_samples('digits')
    X_test = images_test.reshape(images_test.shape[0], 784)
    y_test = labels_test.reshape(-1, 1)
    
    #SCALING
    #scale=np.float32(np.max(X_train))
    scale=255
    X_train =X_train/scale
    X_test=X_test/scale
    
    #STANDARDIZATION
    #pixel by pixel std
    X_train=(X_train-.5)*2
    X_test=(X_test-.5)*2


    X_valid=X_train[:22000,:]
    y_valid=y_train[:22000]

    X_train=X_train[22000:,:]
    y_train=y_train[22000:]
    '''
    X_valid=X_valid.astype(np.float32)
    X_train=X_train.astype(np.float32)
    X_test=X_test.astype(np.float32)
    y_train = y_train.ravel()  # Or labels.reshape(-1)
    y_test = y_test.ravel()
    y_valid=y_valid.ravel()
    '''
    #mnist dataset
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = X.values
    y = y.astype(int).values

    X = ((X / 255.) - .5) * 2
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=10000, random_state=123, stratify=y)
    X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=1./11, random_state=123, stratify=y_temp)
    '''

    
    # optional to free up some memory by deleting non-used arrays:
    #del X_temp, y_temp, X, y
    num_features=X_train.shape[1]
    #scikit learn -> 10 hidden
    #mnist -> 50 hidden
    #emnist -> 100 hidden
    model=NeuralNetMLP(num_features, num_hidden, num_classes=10)

    
    #training
    mse, accs, accs_val=training(model, X_train, y_train, X_valid, y_valid, eta=eta)
    train_time=time.perf_counter()
    print(f"Training + data loading time : {train_time - start_time:.6f} seconds")
    #testing
    acc_test=testing(model, X_test, y_test)
    end_time = time.perf_counter()
    print(f"Testing time : {end_time - train_time:.6f} seconds")
    plt.plot(range(len(accs)), accs, label="Training data")
    plt.plot(range(len(accs_val)), accs_val, label="Validation data")
    plt.title("Accuracy scores for training data and validation data")
    plt.ylabel("Accuracy score")
    plt.xlabel("Epoch")
    plt.legend(loc="lower right")
    plt.show()
    

if __name__=="__main__":
    print("hello")

        

