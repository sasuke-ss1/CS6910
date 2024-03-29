import numpy as np
from FNN import MLP
from dataset import dataset
from sklearn.metrics import confusion_matrix
from argparse import ArgumentParser
from loss import *
import sys
import wandb
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


def get_accuracy(y_pred, y_true):
    a = (np.argmax(y_pred, axis=1)==np.argmax(y_true, axis=1))
    a = a.astype(np.float32)
    return a.mean()


def train(model, dataset, loss, optim, args, confusion =False, log=True):    
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
        if log:
            print(f"EPOCH: {epoch+1}")
        for images, labels in train_data:
            if images.shape[0] == 0:
                continue
            y_pred = model.forward(images)
            train_loss_batch.append(loss(y_pred, labels) + args.weight_decay*model.get_norm())
            accu_train_batch.append(get_accuracy(y_pred, labels))
            model.backward(labels)
            if optim == "nag":
                model.step(labels)
            else:
                model.step()
           

        #print(accu_train_batch[-1])
        
        epoch_train_losses.append(sum(train_loss_batch)/len(train_loss_batch))
        accu_train_epoch.append(sum(accu_train_batch)/len(accu_train_batch))
        if log:
            print(f"Train Epoch Loss: {epoch_train_losses[-1]}")
            print(f"Train Accuracy: {accu_train_epoch[-1]}")

            print("Running Validation")
        if not confusion:
            for images, labels in val_data:
                if images.shape[0] == 0:
                    continue
                y_pred = model.forward(images)
                val_loss_batch.append(loss(y_pred, labels))
                accu_val_batch.append(get_accuracy(y_pred, labels))
                

            
            epoch_val_losses.append(sum(val_loss_batch)/len(val_loss_batch))
            accu_val_epoch.append(sum(accu_val_batch)/len(accu_val_batch))
            if log:
                print(f"Val Epoch Loss: {epoch_val_losses[-1]}")
                print(f"Val Accuracy: {accu_val_epoch[-1]}")
                print("\n\n") 
    if confusion:
        pred =  model.forward(X_val)
        con = confusion_matrix(np.argmax(y_val_one_hot, axis=1), np.argmax(pred, axis=1))

        return con

    return sum(accu_val_epoch)/len(accu_val_epoch)


    


def train_wb():
    loss_dict = {
        "mean_squared_error": MSE(),
        "cross_entropy": CrossEntropy()
    }
    run = wandb.init()
    config = wandb.config
    wandb.run.name = "e_{}_hl_{}_opt_{}_bs_{}_init_{}_ac_{}".format(config.epochs,config.hidden_size,config.optimizer, \
            config.batch_size,config.weight_init,config.activation)  
    class argument:
        def __init__(self):
            self.wandb_project="assignment1"
            self.wandb_entity="sasuke"
            self.dataset="fashion_mnist"
            self.epochs=1
            self.batch_size=4
            self.loss="cross_entropy"
            self.optimizer="sgd"
            self.learning_rate=0.1
            self.momentum=0.5
            self.beta=0.5
            self.beta1=0.5
            self.beta2=0.5
            self.epsilon=0.000001
            self.weight_decay=.0
            self.weight_init="random"
            self.num_layers=1
            self.hidden_size=4
            self.activation="sigmoid"
    args = argument()

    optim_params = {
                "sgd": [config.learning_rate],
                "momentum": [config.learning_rate, args.momentum],
                "nag": [config.learning_rate, args.momentum],
                "rmsprop": [config.learning_rate, args.beta, args.epsilon],
                "adam": [config.learning_rate, args.beta1, args.beta2, args.epsilon],
                "nadam": [config.learning_rate, args.beta1, args.beta2, args.epsilon]
            }

    X_train, X_val, y_train_one_hot, y_val_one_hot , num_classes = dataset(args.dataset)

    Layers = [784];[Layers.append(config.hidden_size) for _ in range(config.num_layers)];Layers.append(num_classes)
    model = MLP(Layers, config.activation, optim=config.optimizer, optim_param = optim_params[config.optimizer], weight_init=config.weight_init, wd=config.weight_decay, loss=config.losses)
    loss = loss_dict[config.losses]

    train_data = list(zip(X_train, y_train_one_hot))
    val_data = list(zip(X_val, y_val_one_hot))

    for epoch in range(config.epochs):
        val_loss_batch = []
        epoch_train_losses = []
        epoch_val_losses = []
        train_loss_batch = []
        accu_train_batch = []
        accu_val_batch = []
        accu_train_epoch= []
        accu_val_epoch = []
        
        for idx, (images, labels) in enumerate(train_data):
            if images.shape[0] == 0:
                continue
            y_pred = model.forward(images)
            train_loss_batch.append(loss(y_pred, labels) + config.weight_decay*model.get_norm())
            accu_train_batch.append(get_accuracy(y_pred, labels))
            model.backward(labels)
            if config.optimizer == "nag":
                model.step(labels)
            else:
                model.step()
        
        epoch_train_losses.append(sum(train_loss_batch)/len(train_loss_batch))
        accu_train_epoch.append(sum(accu_train_batch)/len(accu_train_batch))


        for idx, (images, labels) in enumerate(val_data):
            if images.shape[0] == 0:
                continue
            y_pred = model.forward(images)
            val_loss_batch.append(loss(y_pred, labels))
            accu_val_batch.append(get_accuracy(y_pred, labels))
            

        
        epoch_val_losses.append(sum(val_loss_batch)/len(val_loss_batch))
        accu_val_epoch.append(sum(accu_val_batch)/len(accu_val_batch))

        wandb.log(  
                    {
                        "train_loss": epoch_train_losses[-1],
                        "train_accuracy": accu_train_epoch[-1],
                        "val_loss": epoch_val_losses[-1],
                        "val_accuracy": accu_val_epoch[-1]
                
                    }
                )
    


    
if __name__ == "__main__":
    parser = ArgumentParser()
    ## wandb.ai agg arguments
    parser.add_argument("--wandb_project", "-wp", default="assing1", type=str, help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("--wandb_entity", "-we", default="sasuke", type=str, help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
    parser.add_argument("--dataset", "-d", default="fashion_mnist", type=str, help='choices: ["mnist", "fashion_mnist"]') ## "" in string
    parser.add_argument("--epochs", "-e", default=10, type=int, help="Number of epochs to train neural network.")
    parser.add_argument("--batch_size", "-b", default=512, type=int,  help="Batch size used to train neural network.")
    parser.add_argument("--loss", "-l", default="cross_entropy", type=str, help='choices: ["mean_squared_error", "cross_entropy"]')
    parser.add_argument("--optimizer", "-o", default="nadam", type=str, help='choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]')
    parser.add_argument("--learning_rate", "-lr", default=0.001, type=float, help="Learning rate used to optimize model parameters")
    parser.add_argument("--momentum", "-m", default=0.5, type=float, help="	Momentum used by momentum and nag optimizers.")
    parser.add_argument("--beta", "-beta", default=0.5, type=float, help="Beta used by rmsprop optimizer")
    parser.add_argument("--beta1", "-beta1", default=0.5, type=float, help="Beta1 used by adam and nadam optimizers.")
    parser.add_argument("--beta2", "-beta2", default=0.5, type=float, help="Beta2 used by adam and nadam optimizers.")
    parser.add_argument("--epsilon", "-eps", default=0.000001, type=float, help="Epsilon used by optimizers.")
    parser.add_argument("--weight_decay", "-w_d", default=0.0, type=float, help="Weight decay used by optimizers.")
    parser.add_argument("--weight_init", "-w_i", default="xavier", type=str, help='choices: ["random", "xavier"]')
    parser.add_argument("--num_layers", "-nhl", default=5, type=int, help="Number of hidden layers used in feedforward neural network.")
    parser.add_argument("--hidden_size", "-sz", default=64, type=int, help="Number of hidden neurons in a feedforward layer.")
    parser.add_argument("--activation", "-a", default="relu", type=str, help='choices: ["identity", "sigmoid", "tanh", "relu"]')
    parser.add_argument("--question", "-q", type=int, default=None, help="The Question Number you want to run")
    parser.add_argument("--custom", "-c", type=bool, default=False, help="Create a neural network with differnt number of neurons in each layer")

    loss_dict = {
        "mean_squared_error": MSE(),
        "cross_entropy": CrossEntropy()
    }
    args = parser.parse_args()
    
    data = dataset(args.dataset, batch_size=args.batch_size)
    
    num_classes = 10

    if args.question:
        wandb.login(key="e99813e81e3838e6607d858a20693d589933495f")
        with open("./sweep.yml", "r") as f:
            sweep_config = yaml.safe_load(f)
        
        #question 1
        if args.question == 1:
            wandb.init(project=args.wandb_project)
            wandb.run.name = "question-1"
            X_train, X_val, y_train_one_hot, y_val_one_hot , num_classes = data
            x = np.concatenate(X_train, axis=0);y = np.concatenate(y_train_one_hot, axis=0)
            if args.dataset == "fashion_mnist":
                class_mapping = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"} 
            else:
                class_mapping = {i:str(i) for i in range(10)}

            y = np.argmax(y, axis=0)
            x = x[y]
        
            wandb.log({"Question 1": [wandb.Image(x[i].reshape(28, 28, 1), caption=class_mapping[i]) for i in range(10)]})

        #question 2, 3
        elif args.question in [2,3]:
            print("For Question 2&3 run the train.py again but without specifying any question number")

        #question 4

        elif args.question == 4:
            sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
            wandb.agent(sweep_id, function=train_wb)

        elif args.question in [5, 6]:
            raise ValueError ("Please Check the Readme and my wandb assignment page")

        elif args.question in [7, 10]:
            wandb.init(project=args.wandb_project)
            wandb.run.name = f"question-{args.question}"
            args.dataset = "mnist" if args.question == 10 else "fashion_mnist"

            data = dataset(args.dataset, batch_size=args.batch_size, test=True)            
            Layers = [784];[Layers.append(args.hidden_size) for _ in range(args.num_layers)];Layers.append(num_classes)
            optim_params = {
                "sgd": [args.learning_rate],
                "momentum": [args.learning_rate, args.momentum],
                "nag": [args.learning_rate, args.momentum],
                "rmsprop": [args.learning_rate, args.beta, args.epsilon],
                "adam": [args.learning_rate, args.beta1, args.beta2, args.epsilon],
                "nadam": [args.learning_rate, args.beta1, args.beta2, args.epsilon]
            }

            Model = MLP(Layers = Layers, optim=args.optimizer, optim_param=optim_params[args.optimizer], weight_init = args.weight_init,\
                wd = args.weight_decay, activation=args.activation)
                
            Model.summary()
            
            cm = train(Model, data ,loss_dict["cross_entropy"], args.optimizer, args = args, confusion=True)
            ### Confusion Matrix

            df_cm = pd.DataFrame(cm, index = range(1, len(cm)+1), columns = range(1, len(cm) + 1))
            plt.figure(figsize=(12,12))
            ax = plt.subplot()
            tmp = sn.heatmap(df_cm, annot=True)
            ax.set_xlabel("Predicted Labels");ax.set_ylabel('True Labels')
            ax.set_title("Confusion Matrix")

            if args.dataset == "fashion_mnist":
                class_mapping = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"} 
            else:
                class_mapping = {i:str(i) for i in range(10)}
            
            ax.xaxis.set_ticklabels(list(class_mapping.values()));ax.yaxis.set_ticklabels(list(class_mapping.values()))
            fig = tmp.get_figure()
            wandb.log({f"Question-{args.question}": wandb.Image(fig)})
            fig.savefig(f"./confusion{args.question}.png", dpi=400)
            plt.show()
            
        elif args.question == 8:
            run = wandb.init(project=args.wandb_project)
            wandb.run.name = "question-8"
            optim_params = {
                "sgd": [args.learning_rate],
                "momentum": [args.learning_rate, args.momentum],
                "nag": [args.learning_rate, args.momentum],
                "rmsprop": [args.learning_rate, args.beta, args.epsilon],
                "adam": [args.learning_rate, args.beta1, args.beta2, args.epsilon],
                "nadam": [args.learning_rate, args.beta1, args.beta2, args.epsilon]
            }

            data = dataset("mnist", batch_size=args.batch_size)            

            Layers = [784];[Layers.append(args.hidden_size) for _ in range(args.num_layers)];Layers.append(num_classes)

            Model_2 = MLP(Layers = Layers, optim=args.optimizer, optim_param= optim_params[args.optimizer], weight_init = args.weight_init,\
                wd = args.weight_decay, activation=args.activation, loss="mean_squared_error")

            Model_1 = MLP(Layers = Layers, optim=args.optimizer, optim_param= optim_params[args.optimizer], weight_init = args.weight_init,\
                wd = args.weight_decay, activation=args.activation, loss="cross_entropy")
            val_mse, val_cross = [], []
            epochs = args.epochs
            args.epochs = 1
            for epochs in range(epochs):

                val_mse.append(train(Model_2, data ,loss_dict["mean_squared_error"], args.optimizer, args=args, log=False))
                val_cross.append(train(Model_1, data ,loss_dict["cross_entropy"], args.optimizer, args= args, log=False))
            print(len(val_mse))
            print(len(val_cross))
            wandb.log({
                "mse vs crossentropy": wandb.plot.line_series(
                xs = list(range(1, epochs+1)),
                ys = [val_mse, val_cross],
                keys=["mse", "cross_entropy"],
                title = "mse vs cross_entropy",
                xname = "epochs"
                )
            })
        else:
            raise ValueError("No such Quesition")



    else:
        optim_params = {
            "sgd": [args.learning_rate],
            "momentum": [args.learning_rate, args.momentum],
            "nag": [args.learning_rate, args.momentum],
            "rmsprop": [args.learning_rate, args.beta, args.epsilon],
            "adam": [args.learning_rate, args.beta1, args.beta2, args.epsilon],
            "nadam": [args.learning_rate, args.beta1, args.beta2, args.epsilon],
            "nag_custom_your_optim": [args.learning_rate]
        }
        '''
        To use your own optimizer you need to add the name of the optimizer to the optim_params dictionary with
        the corresponding parameters mapping (Note: The name must contain the substring 'custom'), you will also need to your write your optimizer implementaion in the class 'Your_Optimizer'.
        There you would need to implement the __init__() function and then the __call__() function.
        If your optimizer uses nestrov acceleration then you need to write your name is optimizer name should start with nag.(Note: If you are using nag then please define the momentum parameter as beta)

        If you did everything correctly then you should be able to call your implemntaion my using the optimizer command line input with the name of your optimizer.
        '''
        Layers = [784];[Layers.append(args.hidden_size) for _ in range(args.num_layers)];Layers.append(num_classes)

        if args.custom:
            print(f"Please inputs neuron sizes for the {args.num_layers} hidden_layers by pressing ENTER after every input")
            Layers = [784];[Layers.append(int(input())) for _ in range(args.num_layers)];Layers.append(num_classes)    

        Model = MLP(Layers = Layers, optim=args.optimizer, optim_param= optim_params[args.optimizer], weight_init = args.weight_init, wd = args.weight_decay, activation=args.activation, loss=args.loss)
        Model.summary()

        train(Model, data ,loss_dict[args.loss], args.optimizer, args = args)
