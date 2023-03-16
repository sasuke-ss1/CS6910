import numpy as np
from weights_init import *
from activations import *


class Linear():
    def __init__(self, fin, fout, activation="sigmoid", init="xavier", last=False):
        self.w = None
        if init == "xavier":
            self.w = xavier(fout, fin)
        else:
            self.w = random(fout, fin)

        self.b = np.zeros(fout)

        self.act = activation

        if activation == "sigmoid":
            self.activation = Sigmoid()
        elif activation == "relu":
            self.activation = ReLU()
        elif activation == "identity":
            self.activation = Identity()
        elif activation == "tanh":
            self.activation = Tanh()
        elif not last:
            raise NotImplementedError

        if last:
            self.activation = Softmax()
            self.act = "softmax"

        self.u_w, self.u_b = 0, 0
        self.v_w, self.v_b = 0, 0
        self.t = 1
        self.delta = np.zeros(fout)
        

    def __call__(self, x):
        return self.activation(x@self.w.T+self.b)
