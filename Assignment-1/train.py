import numpy as np
from FNN import MLP
from dataset import dataset
from argparse import ArgumentParser
from loss import *
import sys

epoch_train_losses = []

def train(model, dataset, loss, optim=None):
    train_loss_batch = []
    accu_train_bacth = []
    z = 0
    for idx, (images, labels) in enumerate(dataset):
        y_pred = model.forward(images)
        train_loss_batch.append(loss(y_pred, labels))
        model.backward(labels)
        if optim == "nag":
            model.step(labels)
        else:
            model.step()
        #z+=1
        #if z ==3:  
        #    for i, net in enumerate(model.network):
        #        print(net.delta)
        #        print("\n\n")
        #    print("#######################################################")
        #    for i, out in enumerate(model.outs):
        #        print(out)
        #        print("\n\n")
        

        
        
    
    epoch_train_losses.append(sum(train_loss_batch)/(idx+1))
    print(f"Train Epoch Loss: {epoch_train_losses[-1]}")
        

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
    parser.add_argument("--weight_init", "-w_i", default="random", type=str, help='''choices: ["random", "Xavier"]''')
    parser.add_argument("--num_layers", "-nhl", default=1, type=int, help="	Number of hidden layers used in feedforward neural network.")
    parser.add_argument("--hidden_size", "-sz", default=4, type=int, help="	Number of hidden neurons in a feedforward layer.")
    parser.add_argument("--activation", "-a", default="sigmoid", type=str, help='''	choices: ["identity", "sigmoid", "tanh", "ReLU"]''')

    loss_dict = {
        "mean_squared_error": MSE(),
        "cross_entropy": CrossEntroy()
    }
    args = parser.parse_args()
    
    X_train, X_val, y_train_one_hot, y_val_one_hot , num_classes = dataset(args.dataset)
    
    Layers = [748];[Layers.append(args.hidden_size) for _ in range(args.num_layers)];Layers.append(num_classes)

    optim_params = {
        "sgd": [args.learning_rate],
        "momentum": [args.learning_rate, args.momentum],
        "nag": [args.learning_rate, args.momentum],
        "rmsprop": [args.learning_rate, args.beta, args.epsilon],
        "adam": [args.learning_rate, args.beta1, args.beta2, args.epsilon],
        "nadam": [args.learning_rate, args.beta1, args.beta2, args.epsilon]
    }

    Model = MLP(Layers=[784, 100 ,num_classes], optim=args.optimizer, optim_param= optim_params[args.optimizer])
    Model.summary()
    
    for epoch in range(args.epochs):
        train(Model, zip(X_train, y_train_one_hot),loss_dict[args.loss], args.optimizer)


