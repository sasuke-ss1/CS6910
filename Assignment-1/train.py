import numpy as np
from FNN import MLP
from dataset import dataset
from argparse import ArgumentParser
from loss import *
import sys
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

#wandb.login()
np.random.seed(42)

def get_accuracy(y_pred, y_true):
    a = (np.argmax(y_pred, axis=1)==np.argmax(y_true, axis=1))
    a = a.astype(np.float32)
    return a.mean()


sweep_configuration = {"name": "complete-sweep", "method": "grid"}
sweep_configuration["metric"] = {"name": "loss", "goal": "minimize"}


params = {
    
    "epochs":{"values": [5, 10]},
    "num_layers":{"values": [3, 4, 5]},
    "hidden_size": {"values": [32, 64, 128]},
    "learning_rate": {"values": [1e-3, 1e-4]},
    "optimizer": {"values": ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]},
    "batch_size": {"values": [16, 32, 64]},
    "weight_init": {"values": ["random", "xavier"]},
    "activation": {"values": ["sigmoid", "tanh", "ReLU"]},
    "loss": {"values": ["cross_entropy", "mean_squared_error"]}
    
}

sweep_configuration["parameters"] = params

def train_wb(config=sweep_configuration):

    loss_dict = {
        "mean_squared_error": MSE(),
        "cross_entropy": CrossEntroy()
    }
    val_loss_batch = []
    epoch_train_losses = []
    epoch_val_losses = []
    train_loss_batch = []
    accu_train_batch = []
    accu_val_batch = []
    accu_train_epoch= []
    accu_val_epoch = []

    with wandb.init(config = config):
        config = wandb.init().config

        optim_param = {
        "sgd": [config.learning_rate],
        "momentum": [config.learning_rate, 0.5],
        "nag": [config.learning_rate, 0.5],
        "rmsprop": [config.learning_rate, 0.5, 0.000001],
        "adam": [config.learning_rate, 0.5, 0.5, 0.000001],
        "nadam": [config.learning_rate, 0.5, 0.5, 0.000001]
        }


        wandb.run.name = "e_{}_hl_{}_opt_{}_bs_{}_init_{}_ac_{}_loss_{}".format(config.epochs,config.hidden_size,config.optimizer, \
                                                                    config.batch_size,config.weight_init,config.activation, config.loss)
        
  
        optim = config.optimizer
        loss = loss_dict[config.loss]

        X_train, X_val, y_train_one_hot, y_val_one_hot , num_classes = dataset(args.dataset)

        Layers = [784];[Layers.append(config.hidden_size) for _ in range(config.num_layers)];Layers.append(num_classes)
        model = MLP(Layers, config.activation, optim=optim, optim_param = optim_param[optim], weight_init=config.weight_init) 

        


        train_data = zip(X_train, y_train_one_hot)
        val_data = zip(X_val, y_val_one_hot)

        for epoch in range(config.epochs):
            for idx, (images, labels) in enumerate(train_data):
                y_pred = model.forward(images)
                train_loss_batch.append(loss(y_pred, labels))
                accu_train_batch.append(get_accuracy(y_pred, labels))
                model.backward(labels)
                if optim == "nag":
                    model.step(labels)
                else:
                    model.step()
            epoch_train_losses.append(sum(train_loss_batch)/(idx+1))
            accu_train_epoch.append(sum(accu_train_batch)/(idx+1))
            print(f"Train Epoch Loss: {epoch_train_losses[-1]}")
            print(f"Train Accuracy: {accu_train_batch[-1]}")

            print("Running Validation")

            for idx, (images, labels) in enumerate(val_data):
                y_pred = model.forward(images)
                val_loss_batch.append(loss(y_pred, labels))
                accu_val_batch.append(get_accuracy(y_pred, labels))
            
            epoch_val_losses.append(sum(val_loss_batch)/(idx+1))
            accu_val_epoch.append(sum(accu_val_batch)/(idx+1))
            print(f"Val Epoch Loss: {epoch_val_losses[-1]}")
            print(f"Val Accuracy: {accu_val_epoch[-1]}")

        wandb.log(  
            {
                "train_loss": sum(epoch_train_losses)/(epoch+1),
                "val_loss": sum(epoch_val_losses)/(epoch+1),
                "train_accuracy": sum(accu_train_epoch)/(epoch+1),
                "val_accuracy": sum(accu_val_epoch)/(epoch+1)
        
            }
        )


        

if __name__ == "__main__":
    parser = ArgumentParser()
    ## wandb.ai agg arguments
    parser.add_argument("--wandb_project", "-wp", default="assignment1", type=str, help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("--wandb_entity", "-we", default="ss_sasuke", type=str, help="	Wandb Entity used to track experiments in the Weights & Biases dashboard.")
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
    
    data = dataset(args.dataset)
    

    optim_params = {
        "sgd": [args.learning_rate],
        "momentum": [args.learning_rate, args.momentum],
        "nag": [args.learning_rate, args.momentum],
        "rmsprop": [args.learning_rate, args.beta, args.epsilon],
        "adam": [args.learning_rate, args.beta1, args.beta2, args.epsilon],
        "nadam": [args.learning_rate, args.beta1, args.beta2, args.epsilon]
    }

    #wandb.init(entity=args.wandb_entity, project=args.wandb_project)

###Question-1
    

    #wandb.log({
    #    [wandb.Image(ig) for ig in img]
    #})
###Question-2
    #Model = MLP(Layers=[784, 100 ,num_classes], optim=args.optimizer, optim_param= optim_params[args.optimizer])
    #Model.summary()
###Question-3

###Question-4

    sweep_id = wandb.sweep(sweep_configuration, project = "trail-1")
    wandb.agent(sweep_id, function = train_wb)


    



    #sweep_id = wandb.sweep(sweep=sweep_configuration, project=)

    #for epoch  in range(args.epochs):
    #    train(Model, zip(X_train, y_train_one_hot),loss_dict[args.loss], args.optimizer)


