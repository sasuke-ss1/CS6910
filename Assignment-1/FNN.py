from layers import *
from loss import *
from optimizer import *
import sys

class MLP():
    def __init__(self, Layers, activation = "sigmoid", loss = "cross_entropy", optim="sgd", optim_param = None, weight_init = "xavier", wd = False):
        network = []
        for i in range(len(Layers) - 2):
            network.append(Linear(Layers[i], Layers[i+1], activation=activation, init=weight_init))
        #network.append(Linear(Layers[-2], Layers[-1], activation=activation))
        network.append(Linear(Layers[-2], Layers[-1], last = True, init=weight_init))
        self.network = network

        self.act = activation
        self.opt = optim

        self.wd = float(wd)

        self.loss = CrossEntroy() if loss == "cross_entropy" else MSE()
        self.optim = None
        if optim == "sgd":
            self.optim = SGD(*optim_param)
        elif optim =="momentum": 
            self.optim = Momentum(*optim_param)
        elif optim == "nag":
            self.optim = NAG(*optim_param)
        elif optim == "rmsprop":
            self.optim = RMSProp(*optim_param)
        elif optim == "adam":
            self.optim = Adam(*optim_param)
        elif optim == "nadam":
            self.optim = NAdam(*optim_param)
        else:
            raise NotImplementedError
            

    def summary(self):
        for i, layer in enumerate(self.network):
            print(f"Weight Parameters in layer_{i}: {layer.w.shape}")
            print(f"Bias Parameters in layers_{i}: {layer.w.shape}")
            print(f"Activation of layers_{i}: {layer.act}")
            print("\n\n")


    def forward(self, x):
        tmp = x
        self.outs = [tmp]
        for layer in self.network:
            tmp = layer(tmp)
            self.outs.append(tmp)
        return tmp

    def backward(self, y_true):
        self.network[-1].delta = self.loss.grad(self.outs[-1], y_true).T
        for i in range(len(self.network)-2, -1, -1):
            self.network[i].delta = (self.network[i+1].w.T@self.network[i+1].delta)*self.network[i].activation.grad().T

    def step(self, y_true = None):
        if self.opt == "nag":
            beta = self.optim.beta
            for i,  layer in enumerate(self.network):
                layer.w -= beta*layer.u_w
                layer.b -= beta*layer.u_b

            self.forward(self.outs[0])
            self.backward(y_true)
            
            for i, layer in enumerate(self.network):
                layer.w += beta*layer.u_w
                layer.b += beta*layer.u_b
                self.optim(layer, layer.delta@self.outs[i], np.sum(layer.delta, axis=1))
            
        else:
            for i, layer in enumerate(self.network):
                #print(self.outs[i+1].shape)
                self.optim(layer, layer.delta@self.outs[i] + self.wd*layer.w, np.sum(layer.delta, axis=1))
