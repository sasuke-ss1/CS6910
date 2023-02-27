import numpy as np


class Sigmoid():
    def __call__(self, x):
        self.z = 1/(1+np.exp(-x))
        return self.z

    def grad(self):

        return self.z*(1-self.z)

class ReLU():
    def __call__(self, x):
        self.z = np.maximum(0, x)
        return self.z

    def grad(self):
        return np.where(self.z>0, np.ones(1,), np.zeros(1,))

class Identity():        
    def __call__(self, x):
        self.z = x
        return x

    def grad(self):
        return np.ones_like(self.z)   ###### Check

class Tanh():
    def __call__(self, x):
        self.z = np.tanh(x)
        return self.z
    
    def grad(self):
        return 1-self.z*self.z

class Softmax():
    def __call__(self, x):
        f = np.exp(x-np.max(x, axis=1).reshape(-1,1))
        #print(f[0])
        self.z = f / f.sum(axis=1).reshape(-1,1)
        #print(self.z[0])
        return self.z
    
    def grad(self, y_true):
        pass

class Your_activaiton():
    def __call__(self, x):
        pass

    def grad(self):
        pass

