import numpy as np

def random(fout, fin):
    return 0.1*np.random.randn(fout, fin)

def xavier(fout, fin):
    lim = np.sqrt(6/float(fin+fout))
    return  0.1*np.random.uniform(-lim, lim, size=(fout, fin))


if __name__ == "__main__":
    pass