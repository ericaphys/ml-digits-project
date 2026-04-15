import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, accuracy_score
from numba import jit
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.colors import Normalize

#--------------------------------------------------------------------
#----- this is a logistic regression model with stochastic gd -------
#--------------------------------------------------------------------

def animazione2D(n_frames,frames):
    #psi: array (n_punti,n_frames)
    #x: array(n_punti)

    fig, ax = plt.subplots()
    maxv=-1e10
    minv=+1e10
    for i in range(n_frames):
        val=frames[i].max()
        if(val>maxv): maxv=val
        val=frames[i].min()
        if(val<minv): minv=val
    vmin=0
    vmax=maxv
    ims = ax.imshow(frames[0], animated=True, extent=[0,10, 10, 0], vmax=vmax, vmin=vmin,interpolation='bilinear', cmap=cm.viridis)
     
    def update(n_frames):
        ims.set_array(frames[i])
        return [ims]
    
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin, vmax), cmap=cm.viridis),
        ax=ax, anchor=(1.0, 1.0),format="%4.2e")

    ani = animation.FuncAnimation(fig=fig, func=update, frames=n_frames,blit=True)
    plt.show()

#LOGISTIC REGRESSION
@jit
def activation_function(z):
    dots=1./(1.+np.exp(-np.clip(z,-250.0, 250.0)))
    return dots.astype(np.float32)

@jit
def shuffle_time(X,y):
    r=np.random.permutation(len(y))
    return X[r], y[r]

@jit
def fit_training(X, y_label, w, n_iter, eta, p, shuffle=True):
    samples=len(X)
    predictions = np.zeros( (n_iter, samples), dtype=np.int64)
    
  
    #STOCHASTIC GRADIENT DESCENT    
    true_Y=[]
    avg_loss=np.zeros(n_iter, dtype=np.float32)
    for epoch in range(n_iter):
        tot_loss=0
        if shuffle==True:
            X, y_label = shuffle_time(X,y_label)
            true_Y.append(y_label)
        for i in range(samples):
            xi=X[i]
            #one_hot encoding
            y_targets=np.zeros(p, dtype=np.float32)
            target_id=y_label[i]
            y_targets[target_id]=1.0
                            
            dots=np.dot(w,xi)#perceptrons dot products

            output=activation_function(dots)

            #error computation using stochastic gradient descent
            errors=y_targets-output
            #print(f"errors: {errors.shape}\nxi: {xi.shape}")
            w += eta*(2.0*np.outer(errors, xi))
            #
            #y_list=np.where(dots>0.5, 1.0, 0.0)
            eps = 1e-6
            output = np.clip(output, eps, 1 - eps)
            cost=-np.dot(y_targets,np.log(output))-(np.dot(1-y_targets,np.log(1-output)))
            tot_loss+=cost    
            predictions[epoch,i]=np.argmax(dots)
        
        avg_loss[epoch]=tot_loss/(p*samples)
      
    return w, predictions, true_Y, avg_loss



class Perceptron:
    #eta -> learning rate
    #n_iter -> passes over training set
    #random_state -> random num generator seed for random weights
    def __init__(self, eta=1, n_iter=10):
        self.eta=eta
        self.n_iter=n_iter
        self.shuffle=True
    

    def fit(self, data):
        matr=[]
        '''fit training data
        X= [n_examples, n_features] training vector
        y= [n_examples] target values
        returns an object'''
        self.p=len(np.unique(data[:, 0])) #number of perceptrons depends on individual classes ()

        self.b_ = float(1.) #scalar bias
        #setting weights and bias in place of the target labels 0col
        self.w_ = np.random.uniform(-0.5,0.5,(self.p,len(data[0]))).astype(np.float32) #weights randomly initialized as a matrix [perceptron][features]
        self.w_[:][0]=self.b_

        #scale=np.float32(np.max(data[:,1:]))
        y_label=data[:,0].astype(int) #these are the true labels of the examples
        X=data[:,1:].astype(np.float32) #these are the input data

        
        X=np.insert(X,0,1.0,axis=1)#insert fixed x0=1


        accuracy=[]
        self.w_, all_epochs, true_y, avg_loss= fit_training(X,y_label, self.w_, self.n_iter, self.eta,self.p, self.shuffle)
        #print(loss)
        for epoch in range(self.n_iter):
            #confusion matrix
            print("Confusion matrix for train data for epoch ", str(int(epoch+1)))
            print(confusion_matrix(true_y[epoch],all_epochs[epoch]))
            matr.append(confusion_matrix(true_y[epoch],all_epochs[epoch]))
            accuracy.append(accuracy_score(true_y[epoch], all_epochs[epoch][:]))
            acc=accuracy_score(true_y[epoch], all_epochs[epoch][:])            
            print(f"Loss function: {avg_loss[epoch]}")
            print(f"Accuracy score: {accuracy_score(true_y[epoch], all_epochs[epoch][:])}")
            print("---------------------------------------------")                      
        #print(self.w_.shape)
        #animazione2D(self.n_iter,matr)
        #plt.show()
        pixels=int(np.sqrt(len(self.w_[0])-1))
        number=[]
        conta=0
        fig, ax =plt.subplots(4,3)
        for i in range(self.p):
            number.append(self.w_[i,1:].reshape(pixels,pixels))
        for i in range(4):
            for j in range(3):
                if conta<self.p:
                    ax[i,j].imshow(number[conta], cmap='viridis', interpolation='nearest')
                    conta+=1
        ax[3,1].axis('off') 
        ax[3,2].axis('off')
        plt.text(0, 0, f"Accuracy score: {acc:.3f}", va='top')
        plt.suptitle(f"Weights after training with logistic regression, eta: {self.eta}, epochs: {self.n_iter}")
        plt.show()
        return self
    
    
    def test(self, data):
        #separate input data from labels
        y_label=data[:,0].astype(int) #these are the true labels of the examples
        scale=np.float32(np.max(data[:,1:]))
        X=data[:,1:].astype(float)/scale #these are the input data
        
        X=np.insert(X,0,1.0,axis=1)#insert fixed x0=1
        actual_list=[]
        pred_list=[]


        for i in range(len(data)): #iterates over each example
            xi= X[i]    #example is the i row
            dots=np.dot(self.w_,xi)#perceptrons dot products
            actual_list.append(y_label[i])
            pred_list.append(np.argmax(dots))
            #confusion matrix
        print("-------------------------------------------------")
        print("Confusion matrix for test data ")
        print(confusion_matrix(actual_list,pred_list))
        acc=accuracy_score(actual_list, pred_list)
        print(f"Accuracy score: {acc}")
        matr=confusion_matrix(actual_list,pred_list)
        vmin=0
        vmax=matr.max()
        #plt.imshow(matr, cmap=cm.viridis, vmin=vmin, vmax=vmax)
        #plt.show()
        return self



