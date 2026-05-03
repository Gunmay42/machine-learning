import numpy as np
X=np.array([1,2,3,4,5])
y=np.array([2,4,6,8,10])

w=0
b=0
alpha=0.01

echos=1000
def predict(X,b,w):
    return w*X+b

def mse_loss(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)

def gradient_descent(X,y,b,w,alpha):
    n=len(X)
    y_pred=predict(X,w,b)
    dw=(1/n)*np.sum(y_pred-y)*X
    db=(1/n)*np.sum(y_pred-y)
    w=w-alpha*dw
    b=b-alpha*db
    return b,w

for epochs in range(echos):
    y_pred=predict(X,w,b)
    loss=mse_loss(y,y_pred)
    b,w=gradient_descent(X,y,b,w,alpha)
    if epochs%100==0:
        print(f"Epoch {epochs}, Loss: {loss}")