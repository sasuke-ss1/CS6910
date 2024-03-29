import numpy as np
class MSE():
    def __call__(self, y_pred, y_true):
        return np.mean((y_pred-y_true)**2)

    def grad(self, y_pred, y_true):
        return 2*(y_pred - y_true)*(y_pred * (1-y_pred))/(y_true.shape[0]*y_true.shape[1])

class CrossEntropy():
    def __call__(self, y_pred, y_true):
        return np.mean(np.sum((-1.0 * np.multiply(y_true, np.log(y_pred))), axis=1))

    def grad(self, y_pred, y_true):
       
        grad = -(1/y_true.shape[0])*(y_true -y_pred)
        #print(grad[0])

        return grad
