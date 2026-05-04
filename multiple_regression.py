import numpy as np
from sklearn.datasets import load_diabetes

data=load_diabetes()
X,y=data.data,data.target
X=(X-X.mean(axis=0))/X.std(axis=0) #Feature scaling

w=np.zeros(X.shape[1]) #Initialize weights for each feature and vectorizing w
b=0
alpha=0.01
epochs=5000

def predict(X,w,b):
    return np.dot(X,w)+b

def mse_loss(y_true,y_pred):
    return (np.mean((y_true-y_pred)**2))

def gradient_descent(X,y,w,b,alpha):
    n=len(X)
    y_pred=predict(X,w,b)
    dw=1/n*np.dot(X.T,y_pred-y) #Vectorized dw
    db=1/n*np.sum(y_pred-y)
    w=w-alpha*dw
    b=b-alpha*db
    return w,b 

for epochs in range(epochs):
    y_pred= predict(X,w,b)
    loss=mse_loss(y,y_pred)
    w,b=gradient_descent(X,y,w,b,alpha)
    if epochs%100==0:
        print(f"Epoch {epochs},Loss:{loss}")
