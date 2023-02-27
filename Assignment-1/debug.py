import numpy as np
from FNN import MLP
from dataset import dataset
from argparse import ArgumentParser
from loss import *
import sys


def get_accuracy(y_pred, y_true):
    a = (np.argmax(y_pred, axis=1)==np.argmax(y_true, axis=1))
    a = a.astype(np.float32)
    return a.mean()


def train(model, dataset, loss, optim=None, args=None):    
    X_train, X_val, y_train_one_hot, y_val_one_hot , num_classes = dataset

    train_data = list(zip(X_train, y_train_one_hot))
    val_data = list(zip(X_val, y_val_one_hot))

    for epoch in range(args.epochs):
        val_loss_batch = []
        epoch_train_losses = []
        epoch_val_losses = []
        train_loss_batch = []
        accu_train_batch = []
        accu_val_batch = []
        accu_train_epoch= []
        accu_val_epoch = []
        print(F"EPOCH: {epoch+1}")
        for idx, (images, labels) in enumerate(train_data):
            if images.shape[0] == 0:
                continue
            y_pred = model.forward(images)
            train_loss_batch.append(loss(y_pred, labels))
            accu_train_batch.append(get_accuracy(y_pred, labels))
            model.backward(labels)
            if optim == "nag":
                model.step(labels)
            else:
                model.step()
           

        #print(accu_train_batch[-1])
        
        epoch_train_losses.append(sum(train_loss_batch)/len(train_loss_batch))
        accu_train_epoch.append(sum(accu_train_batch)/len(accu_train_batch))
        print(f"Train Epoch Loss: {epoch_train_losses[-1]}")
        print(f"Train Accuracy: {accu_train_epoch[-1]}")

        print("Running Validation")

        for idx, (images, labels) in enumerate(val_data):
            if images.shape[0] == 0:
                continue
            y_pred = model.forward(images)
            val_loss_batch.append(loss(y_pred, labels))
            accu_val_batch.append(get_accuracy(y_pred, labels))
            

        
        epoch_val_losses.append(sum(val_loss_batch)/len(val_loss_batch))
        accu_val_epoch.append(sum(accu_val_batch)/len(accu_val_batch))
        print(f"Val Epoch Loss: {epoch_val_losses[-1]}")
        print(f"Val Accuracy: {accu_val_epoch[-1]}")
        print("\n\n")
 
                

    
        
    
    
        

if __name__ == "__main__":
    parser = ArgumentParser()
    ## wandb.ai agg arguments
    parser.add_argument("--wandb_project", "-wp", default="assignment1", type=str, help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("--wandb_entity", "-we", default="sasuke", type=str, help="	Wandb Entity used to track experiments in the Weights & Biases dashboard.")
    parser.add_argument("--dataset", "-d", default="fashion_mnist", type=str, help='''choices: ["mnist", "fashion_mnist"]''') ## "" in string
    parser.add_argument("--epochs", "-e", default=1, type=int, help="	Number of epochs to train neural network.")
    parser.add_argument("--batch_size", "-b", default=4, type=int,  help="Batch size used to train neural network.")
    parser.add_argument("--loss", "-l", default="cross_entropy", type=str, help='''choices: ["mean_squared_error", "cross_entropy"]''')
    parser.add_argument("--optimizer", "-o", default="sgd", type=str, help='''choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]''')
    parser.add_argument("--learning_rate", "-lr", default=0.1, type=float, help="Learning rate used to optimize model parameters")
    parser.add_argument("--momentum", "-m", default=0.5, type=float, help="	Momentum used by momentum and nag optimizers.")
    parser.add_argument("--beta", "-beta", default=0.5, type=float, help="Beta used by rmsprop optimizer")
    parser.add_argument("--beta1", "-beta1", default=0.5, type=float, help="	Beta1 used by adam and nadam optimizers.")
    parser.add_argument("--beta2", "-beta2", default=0.5, type=float, help="	Beta2 used by adam and nadam optimizers.")
    parser.add_argument("--epsilon", "-eps", default=0.000001, type=float, help="Epsilon used by optimizers.")
    parser.add_argument("--weight_decay", "-w_d", default=.0, type=float, help="	Weight decay used by optimizers.")
    parser.add_argument("--weight_init", "-w_i", default="random", type=str, help='''choices: ["random", "xavier"]''')
    parser.add_argument("--num_layers", "-nhl", default=1, type=int, help="	Number of hidden layers used in feedforward neural network.")
    parser.add_argument("--hidden_size", "-sz", default=4, type=int, help="	Number of hidden neurons in a feedforward layer.")
    parser.add_argument("--activation", "-a", default="sigmoid", type=str, help='''	choices: ["identity", "sigmoid", "tanh", "ReLU"]''')

    loss_dict = {
        "mean_squared_error": MSE(),
        "cross_entropy": CrossEntropy()
    }
    args = parser.parse_args()
    
    data = dataset(args.dataset, batch_size=args.batch_size)
    
    num_classes = 10
    #Layers = [784];[Layers.append(args.hidden_size) for _ in range(args.num_layers)];Layers.append(num_classes)

    optim_params = {
        "sgd": [args.learning_rate],
        "momentum": [args.learning_rate, args.momentum],
        "nag": [args.learning_rate, args.momentum],
        "rmsprop": [args.learning_rate, args.beta, args.epsilon],
        "adam": [args.learning_rate, args.beta1, args.beta2, args.epsilon],
        "nadam": [args.learning_rate, args.beta1, args.beta2, args.epsilon]
    }

    Model = MLP(Layers=[784, 64, 32 , 10], optim=args.optimizer, optim_param= optim_params[args.optimizer], weight_init = args.weight_init)
    Model.summary()
    
    train(Model, data ,loss_dict[args.loss], args.optimizer, args = args)

