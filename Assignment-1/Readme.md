# Assignment 1

In this assignment we need to implement a feedforward neural network and write the backpropagation code for training the network. This network was then tested on Fashion-MNIST and MNIST datasets.

## Training

To train this network we run the command 

```sh
python train.py
```
| Name                   | Default Value                    | Description                     |
| :--------------------: | :------------------------------: | ------------------------------- |
| ``` -wp ```, ``` --wandb_project ```  | assignment1   |Project name used to track experiments in Weights & Biases dashboard
| ``` -we ```, ``` --wandb_entity ```   | sasuke        |Wandb Entity used to track experiments in the Weights & Biases dashboard
| ``` -d ```, ``` --dataset ```         | fashion_mnist |Choices: ["mnist", "fashion_mnist"]
| ``` -e ```, ``` --epochs ```          | 1             |Number of epochs to train neural network
| ``` -b ```, ``` --batch_size ```      | 4             |Batch size used to train neural network
| ``` -l ```, ``` --loss ```            | cross_entropy |Choices: ["mean_squared_error", "cross_entropy"]
| ``` -o ```, ``` --optimizer ```       | sgd           |Choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]
| ``` -lr ```, ``` --learning_rate ```  | 0.1           |Learning rate used to optimize model parameters
| ``` -m ```, ``` --momentum ```        | 0.5           |Momentum used by momentum and nag optimizers
| ``` -beta ```, ``` --beta ```         | 0.5           |Beta used by rmsprop optimizer
| ``` -beta1 ```, ``` --beta1 ```       | 0.5           |Beta1 used by adam and nadam optimizers
| ``` -beta2 ```, ``` --beta2 ```       | 0.5           |Beta2 used by adam and nadam optimizers
| ``` -eps ```, ``` --epsilon ```       | 0.000001      |Epsilon used by optimizers
| ``` -w_d ```, ``` --weight_decay ```  | .0            |Weight decay used by optimizers
| ``` -w_i ```, ``` --weight_init ```   | random        |Choices: ["random", "xavier"]
| ``` -nhl ```, ``` --num_layers ```    | 1             |Number of hidden layers used in feedforward neural network
| ``` -sz ```, ``` --hidden_size ```    | 4             |Number of hidden neurons in a feedforward layer
| ``` -a ```, ``` --activation ```      | sigmoid       |Choices: ["identity", "sigmoid", "tanh", "relu"]
| ``` -q ```, ``` --question ```        | None          |Question Number of Assignment

Note: If Question number is specified the ``` train.py ``` will run the code for the intended question if someone has to try some random configuration then they can runt ``` python train.py ``` with the required command line arguments specified


## Question 1

The program, reads the data from ``` keras.datasets ```, picks one example from each class and logs the same to ``` wandb ``` the user can change the dataset whoes samples are being plotted by changing the command line argument.

## Question 2

The Model is implemented in the ``` FNN.py ``` the Model class namely ``` MLP ``` has 5 instance methods namely ``` __init__() ```, ```s summary() ```, ```  forward() ```, ``` backward() ```, ``` step() ```.

The ``` __init__() ``` is the main function that is called when the class object is instantiated which is responsible for bulding the layers of the model, giving them activation functions, optimizer, weight initiailizations, weight decay coefficient for L2 regularization and finally the loss fucntion.

All the layers are Affine with some activation function attached to them, the code for this is in the ``` layers.py ``` file. The ``` summary() ``` function prints the architecture of the neural network.

This basically implements $a(w^{T}x + b)$ where $a(\cdot)$ is the activation function assigned to the layer.
The ``` forward() ``` function defines the forward pass throughout the whole network and stores the outputs in an instance variable(list) and returns the final output of the netowork.

(Note: It is assumed that the network will only be used for classification so the last layer of the neural netowork will always have __softmax()__ activation.)

The ```sh backward() ``` function implements the backward pass throught the entire network by propagating the gradients backward and calculating the chain rule, finally the ``` step() ``` function will apply the gradient of each layer to that layer sequentially the using the specified the optimizer.

## Question 3

The optimizers are implemented as __class__ in ``` optimizer.py ``` every optimizer has only ``` __call__() ``` defined, which takes in the gradients and layers weights of a particular layer and updates the weights according to the update rule of the particular class.

For implementing new optimizer the user just needs to implement the ``` __call__() ``` function, if the optimizer needs gradient at time step t+1 then the user needs set self.opt as nag !!!!!Edit!!!!!. 


SGD: 
```python
class SGD():
    def __init__(self, lr):
        self.lr = lr

    def __call__(self, layer, grad_w, grad_b):
        #print(layer.w.shape)
        layer.w -= self.lr*grad_w
        layer.b -= self.lr*grad_b
```

Momentum:
```python
class Momentum():
    def __init__(self, lr, momentum):
        self.lr = lr
        self.momentum = momentum

    def __call__(self, layer, grad_w, grad_b):

        layer.u_w = self.momentum*layer.u_w + grad_w
        layer.w -= self.lr*layer.u_w

        layer.u_b = self.momentum*layer.u_b + grad_b
        layer.b -= self.lr*layer.u_b
```

NAG:
```python
class NAG():
    def __init__(self, lr, momentum):
        self.beta = momentum
        self.lr = lr

    def __call__(self, layer, grad_w, grad_b):
        layer.u_w = self.beta*layer.u_w + grad_w
        layer.w -= self.lr*layer.u_w

        layer.u_b = self.beta*layer.u_b + grad_b
        layer.b -= self.lr*layer.u_b
```

RMSProp:
```python
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
```

Adam:
```python
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
```

NAdam:
```python
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
```
## Question 4
In this question we tried to find the best set of hyperparameters with the help of ``` wandb.sweep() ``` functionality. To use this we first had to specify the strategy and also the metric we are trying to maximize/minimize, we also need to define our search space.

Wandb's sweep function runs a function defined by the user and logs the quantities the user desires. It basically uses multiple agents to run the user defined function for different configuration extracted for specified user configuration space.

We had make a ``` sweep.yml ``` file to specify all the paramaters for wandb and then we defined a ``` train_wb() ``` function which is used by wandb for hyperparameter search.
!!!!!! Stratergy and stuff left !!!!!!!!!

The generated plots from wandb can be found in the !!!!!!!reports pdf!!!!!!!!!!

## Question 5

This is basically the plot of all model vs the accuracies they got, then plots can ne found in the !!!!!reports pdf.!!!!!!

## Question 6

!!!!!!! Can do it right now!!!!!!!

## Question 7

Here we plot the confusion matrix, for this we used the __confusion_matrix__ function from __sklearn__ library,
we plot this for the best model found through hyperparameter search in wandb.

!!!!!!!! As we can see that !!!!!!!!

## Quesiton 8

In this question we compare the __Mean squared error__ with the __Cross entropy__ error empericaly, we see that both of them perform quite identically but cross entropy on average is able to perform marginally better that mean squared error !!!!!! Add more and image!!!!.

## Question 9 and 10
