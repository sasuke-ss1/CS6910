import numpy as np


class SGD():
    def __init__(self, lr):
        self.lr = lr

    def __call__(self, layer, grad_w, grad_b):
        #print(layer.w.shape)
        layer.w -= self.lr*grad_w
        layer.b -= self.lr*grad_b
        

class Momentum():
    def __init__(self, lr, momentum):
        self.lr = lr
        self.momentum = momentum

    def __call__(self, layer, grad_w, grad_b):

        layer.u_w = self.momentum*layer.u_w + grad_w
        layer.w -= self.lr*layer.u_w

        layer.u_b = self.momentum*layer.u_b + grad_b
        layer.b -= self.lr*layer.u_b


class NAG():
    def __init__(self, lr, momentum):
        self.beta = momentum
        self.lr = lr

    def __call__(self, layer, grad_w, grad_b):
        layer.u_w = self.beta*layer.u_w + grad_w
        layer.w -= self.lr*layer.u_w

        layer.u_b = self.beta*layer.u_b + grad_b
        layer.b -= self.lr*layer.u_b
        

class RMSProp():
    def __init__(self, lr, beta, eps):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        
    def __call__(self, layer,grad_w, grad_b):
        layer.v_w = self.beta*layer.v_w + (1-self.beta)*grad_w**2
        layer.w -= self.lr*grad_w/(np.sqrt(layer.v_w) + self.eps)

        layer.v_b = self.beta*layer.v_b + (1-self.beta)*grad_b**2
        layer.b -= self.lr*grad_b/(np.sqrt(layer.v_b) + self.eps)

class Adam():
    def __init__(self, lr, beta1, beta2, eps):
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.eps = eps

    def __call__(self, layer, grad_w, grad_b):
        layer.u_w = self.beta1*layer.u_w + (1-self.beta1)*grad_w
        layer.u_b = self.beta1*layer.u_b + (1-self.beta1)*grad_b

        layer.v_w = self.beta2*layer.v_w + (1-self.beta2)*(grad_w**2)
        layer.v_b = self.beta2*layer.v_b + (1-self.beta2)*(grad_b**2)

        m_hat_w = layer.u_w/(1 - self.beta1**layer.t)
        v_hat_w = layer.v_w/(1 - self.beta2**layer.t)

        m_hat_b = layer.u_b/(1 - self.beta1**layer.t)
        v_hat_b = layer.v_b/(1 - self.beta2**layer.t)

        layer.w -= self.lr*m_hat_w/(np.sqrt(v_hat_w)+self.eps)
        layer.b -= self.lr*m_hat_b/(np.sqrt(v_hat_b)+self.eps)

        layer.t += 1


class NAdam():
    def __init__(self, lr, beta1, beta2, eps):
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.eps = eps

    def __call__(self, layer, grad_w, grad_b):
        
        layer.u_w = self.beta1*layer.u_w + (1-self.beta1)*grad_w
        layer.v_w = self.beta2*layer.v_w + (1-self.beta2)*grad_w**2 

        m_hat_w = layer.u_w / (1 - self.beta1**layer.t)
        v_hat_w = layer.v_w / (1 - self.beta2**layer.t)

        layer.u_b = self.beta1*layer.u_b + (1-self.beta1)*grad_b
        layer.v_b = self.beta2*layer.v_b + (1-self.beta2)*grad_b**2 

        m_hat_b = layer.u_b / (1 - self.beta1**layer.t)
        v_hat_b = layer.v_b / (1 - self.beta2**layer.t)        

        layer.w -= self.lr/(np.sqrt(v_hat_w)+self.eps)*(self.beta1*m_hat_w+(1-self.beta1)*grad_w/(1 - self.beta1**layer.t))
        layer.b -= self.lr/(np.sqrt(v_hat_b)+self.eps)*(self.beta1*m_hat_b+(1-self.beta1)*grad_b/(1 - self.beta1**layer.t))


        layer.t += 1

