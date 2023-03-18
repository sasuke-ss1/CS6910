from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split
import numpy as np

def dataset(name, num_features = 784, batch_size = 128, test=False):
    if name == "mnist":
        (X, y) , (X_test, y_test) = mnist.load_data()
    elif name == "fashion_mnist":
        (X, y) , (X_test, y_test) = fashion_mnist.load_data()
    X
    X = X.reshape(X.shape[0], num_features)/255.0
    X_test = X_test.reshape(X_test.shape[0], num_features)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
### Copied
    M = X_train.shape[0]

    Mval = X_val.shape[0]

    Mtest = X_test.shape[0]

    num_features = 784

    num_classes = len(np.unique(y_train))

    y_train_one_hot = np.zeros((10, M))
    y_train_one_hot[y_train, np.array(list(range(M)))] = 1
    y_train_one_hot = y_train_one_hot.T

    y_val_one_hot = np.zeros((10, Mval))
    y_val_one_hot[y_val, np.array(list(range(Mval)))] = 1
    y_val_one_hot = y_val_one_hot.T

    y_test_one_hot = np.zeros((10, Mtest))
    y_test_one_hot[y_test, np.array(list(range(Mtest)))] = 1
    y_test_one_hot = y_test_one_hot.T

    batch_size = X_train.shape[0]//batch_size + 1
    
    X_train = np.array_split(X_train, batch_size)
    y_train_one_hot = np.array_split(y_train_one_hot, batch_size)

    X_val = np.array_split(X_val, batch_size)
    y_val_one_hot = np.array_split(y_val_one_hot, batch_size)    

    if test:
        return X_train, X_test, y_train_one_hot, y_test_one_hot, num_classes

    return X_train, X_val, y_train_one_hot, y_val_one_hot , num_classes

if __name__ == "__main__":
    #dataset("mnist")
    X_train, X_val, y_train_one_hot, y_val_one_hot , num_classes = dataset("fashion_mnist")

    dat = zip(X_train, y_train_one_hot)
    for (img, label) in dat:
        print(img.shape, label.shape)
        break   