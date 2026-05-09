import numpy as np
from sklearn.datasets import load_breast_cancer

data=load_breast_cancer()
X,y=data.data,data.target

X=(X-X.mean(axis=0))/X.std(axis=0)
w=np.zeros(X.shape[1])
b=0
alpha=0.01
epochs=1000

def sigmoid(z):
    return 1/(1+np.exp(-z))

def pred(X,w,b):
    z=np.dot(X,w)+b
    return sigmoid(z)

def loss(y_true,y_pred):
    m=y_true.shape[0]
    cost=-np.sum(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))/m
    return cost

def gradient_descent(X,y,w,b,alpha):
    m=X.shape[0]
    y_pred=pred(X,w,b)
    dw=(1/m)*np.dot(X.T,(y_pred-y))
    db=1/m*np.sum(y_pred-y)
    w=w-alpha*dw
    b=b-alpha*db
    return w,b 
for epochs in range(epochs):
    y_pred= pred(X,w,b)
    cost=loss(y,y_pred)
    w,b=gradient_descent(X,y,w,b,alpha)
    if epochs%100==0:
        print(f"Epoch {epochs},Loss: {cost}")

y_pred=pred(X,w,b)
y_pred_label=(y_pred>=0.5).astype(int)
accuracy= np.mean(y_pred_label==y)*100
print(f"Accuracy: {accuracy:.2f}%")