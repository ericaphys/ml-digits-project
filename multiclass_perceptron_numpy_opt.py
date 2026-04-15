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



def animazione2D(n_frames,frames):
    #psi: array (n_punti,n_frames)
    #x: array(n_punti)

    fig, ax = plt.subplots()
        #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        #im=ax.plot_surface(X, Y, frames[0], vmin=frames[0].min() * 2, cmap=cm.Blues)


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



@jit
def shuffle_time(X,y):
    r=np.random.permutation(len(y))
    return X[r], y[r]


@jit
def fit_training(X, y_label, w, n_iter, eta, p, shuffle):
    samples=len(X)
    predictions = np.zeros( (n_iter, samples), dtype=np.int64)
    true_Y=[]
    for epoch in range(n_iter):
        if shuffle==True:
            X, y_label = shuffle_time(X,y_label)
            true_Y.append(y_label)
        for i in range(samples):
            xi=X[i]
            y_targets=np.zeros(p)
            target_id=y_label[i]
            y_targets[target_id]=1.0
                            
            dots=np.dot(w,xi)#perceptrons dot products

            y_list=np.where(dots>0, 1.0, 0.0)
            updates=y_targets-y_list

            w+= eta*np.outer(updates,xi)
            
            predictions[epoch,i]=np.argmax(dots)
    
    return w, predictions, true_Y




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
        #scale=np.float32(np.max(data[:,1:]))
        self.b_ = float(1.) #scalar bias
        #setting weights and bias in place of the target labels 0col
        self.w_ = np.random.uniform(-0.5,0.5,(self.p,len(data[0]))).astype(np.float32) #weights randomly initialized as a matrix [perceptron][features]
        self.w_[:][0]=self.b_

        y_label=data[:,0].astype(int) #these are the true labels of the examples
        X=data[:,1:].astype(np.float32) #these are the input data
        
        X=np.insert(X,0,1.0,axis=1)#insert fixed x0=1
        
        
        accuracy=[]
        self.w_, all_epochs, true_y = fit_training(X,y_label, self.w_, self.n_iter, self.eta,self.p, self.shuffle)
       


        for epoch in range(self.n_iter):
            '''
            #errors=0
            actual_list=[]
            pred_list=[]
            for i in range(len(data)): #iterates over each example
                xi= X[i]#example is the i row
                y_list=[]
                y_targets=np.zeros(self.p)
                target_id=y_label[i]
                y_targets[target_id]=1.0
                actual_list.append(y_label[i])
                
                dots=np.dot(self.w_,xi)#perceptrons dot products

                y_list=(dots>0).astype(float)
                updates=y_targets-y_list

                self.w_ += self.eta*np.outer(updates,xi)

                pred_list.append(np.argmax(dots))
            '''
            #confusion matrix
            print("Confusion matrix for train data for epoch ", str(int(epoch+1)))
            print(confusion_matrix(true_y[epoch],all_epochs[epoch]))
            matr.append(confusion_matrix(true_y[epoch],all_epochs[epoch]))
            accuracy.append(accuracy_score(true_y[epoch], all_epochs[epoch][:]))
            acc=accuracy_score(true_y[epoch], all_epochs[epoch][:])
            print(f"Accuracy score: {accuracy_score(true_y[epoch], all_epochs[epoch][:])}")
            print("---------------------------------------------")            
            #print(self.w_)
        #animazione2D(self.n_iter,matr)
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
        plt.suptitle(f"Weights after training with Perceptrons, eta: {self.eta}, epochs: {self.n_iter}")
        plt.text(0, 0, f"Accuracy score: {acc}", va='top')
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
        print("confusion matrix for test data ")
        print(confusion_matrix(actual_list,pred_list))
        acc=accuracy_score(actual_list, pred_list)
        print(f"Accuracy score: {acc}")
        matr=confusion_matrix(actual_list,pred_list)
        vmin=0
        vmax=matr.max()
        #plt.imshow(matr, cmap=cm.viridis, vmin=vmin, vmax=vmax)
        #plt.show()
        return self



