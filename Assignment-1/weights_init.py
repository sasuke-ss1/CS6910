import numpy as np

def random(fout, fin):
    return 0.1*np.random.randn(fout, fin) ##Standard Gaussian

def xavier(fout, fin):
    lim = np.sqrt(2/float(fin+fout))
    return  0.1*np.random.normal(0.0, lim, size=(fout, fin))


if __name__ == "__main__":
    pass