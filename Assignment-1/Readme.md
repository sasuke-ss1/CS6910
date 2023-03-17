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
| ``` -e ```, ``` --epochs ```          | 10            |Number of epochs to train neural network
| ``` -b ```, ``` --batch_size ```      | 128           |Batch size used to train neural network
| ``` -l ```, ``` --loss ```            | cross_entropy |Choices: ["mean_squared_error", "cross_entropy"]
| ``` -o ```, ``` --optimizer ```       | nadam         |Choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]
| ``` -lr ```, ``` --learning_rate ```  | 0.001         |Learning rate used to optimize model parameters
| ``` -m ```, ``` --momentum ```        | 0.5           |Momentum used by momentum and nag optimizers
| ``` -beta ```, ``` --beta ```         | 0.5           |Beta used by rmsprop optimizer
| ``` -beta1 ```, ``` --beta1 ```       | 0.5           |Beta1 used by adam and nadam optimizers
| ``` -beta2 ```, ``` --beta2 ```       | 0.5           |Beta2 used by adam and nadam optimizers
| ``` -eps ```, ``` --epsilon ```       | 0.000001      |Epsilon used by optimizers
| ``` -w_d ```, ``` --weight_decay ```  | 0.0005        |Weight decay used by optimizers
| ``` -w_i ```, ``` --weight_init ```   | random        |Choices: ["random", "xavier"]
| ``` -nhl ```, ``` --num_layers ```    | 5             |Number of hidden layers used in feedforward neural network
| ``` -sz ```, ``` --hidden_size ```    | 64            |Number of hidden neurons in a feedforward layer
| ``` -a ```, ``` --activation ```      | relu          |Choices: ["identity", "sigmoid", "tanh", "relu"]
| ``` -q ```, ``` --question ```        | None          |Question Number of Assignment
| ``` -c ```, ``` --custom ```          | False         |Create a neural network with differnt number of neurons in each layer


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

The ``` backward() ``` function implements the backward pass throught the entire network by propagating the gradients backward and calculating the chain rule, finally the ``` step() ``` function will apply the gradient of each layer to that layer sequentially the using the specified the optimizer.

There is a __custom__ flag that can be set true through command line, this allows the user to add different numbers of neurons to each hidden layer.
(Note: The use still needs to specify the number of hidden_layers through command line inputs or else the model will use the default values.)

To use your own optimizer you need to add the name of the optimizer to the optim_params dictionary with the corresponding parameters mapping (Note: The name must contain the substring 'custom'), you will also need to your write your optimizer implementaion in the class 'Your_Optimizer'.

There you would need to implement the ``` __init__() ``` function and then the ``` __call__() ``` function.
If your optimizer uses nestrov acceleration then you need to write your name is optimizer name should start with nag.(Note: If you are using nag then please define the momentum parameter as beta)

If you did everything correctly then you should be able to call your implemntaion my using the optimizer command line input with the name of your optimizer.

The code is felxible for all batch sizes, the batch_size can be set using command line arguments.

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
Since Nag is based of "look before you leap" we have define a different step loop nag which is:  
```python 
if self.opt.startswith("nag"):
    beta = self.optim.beta
    for i,  layer in enumerate(self.network):
        layer.w -= beta*layer.u_w
        layer.b -= beta*layer.u_b

    self.forward(self.outs[0])
    self.backward(y_true)
    
    for i, layer in enumerate(self.network):
        layer.w += beta*layer.u_w
        layer.b += beta*layer.u_b
        self.optim(layer, layer.delta@self.outs[i] + self.wd*layer.w, np.sum(layer.delta, axis=1))
 
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

I first tried using grid search stratergy for wandb but quickly realized that it has exponnetial time complextiy and it will take forever to run through all the possible choice of hyperparameters, so from there after analyzing the results of grid search I reduced my hyperparameter search space and used bayesian search stratergy.

I choose bayesian because it actually considers the past history of hyperparameters(prior) selected before selecting the new set of hyperparameters.

So at every run it does the following:
- Choose a set of hyperparameter values (our belief) and use them to train the machine learning model.
- Evalute the performace of the model on those hyperparamters.
- Update our belief accordingly.

So to summarize we begin with an initial estimate of the hyperparameters and progressively update it in light of previous findings.

Note:(My hyperparameter search space can be found in ``` sweep.yml ```)

The generated plots from wandb for both grid and bayes search can be found in the report submitted.

## Question 5

This is basically the plot of all model vs the accuracies they got, then plots can also be found in the report submitted, we see from this graph that bayesian search was able to identify the optimal hyperparameters quite quickly and many model runs results in more 85% validation accuracy.

My best validation set accuracy was __88.24%__ across all the runs.

## Question 6

- We see form the plot of question 5 that initially the model performance is not very good but as time passes it gets better and better, this behaviour is expected form bayesian search as it updates its belief of good hyperparametes with every next run and hence gets better at giving the optimal hyperparameters.
- Most of the runs that resulted in 60-70% accuracy could be blamed on the optimizer as most of them are either using nag, sgd or momentum which is using learning_rate 0.001 or 0.0001 and this might just not be large enough for them to converge.
- The lower accuracy can also be attributed to the tanh and softmax activation functions as they saturate fast and this leads to diminishing gradients flowing back and hence limited updates. This problems only gets worse with the depth of the network. Although the top 40 runs are dominated by relu we still see that tanh is simetimes able to match the accuracy of relu, this maybe because of how simple fashion MNIST dataset is.
- We also wee that larger batch_size models are  in general giving low accuracy than their lower batch_size counterpart. This is beacuse the lower batch_size runs are doing more numbers of updates per epochs than the higher batch_size runs and hence in a way converge faster, this doesnt necessarily means that higher batch_size is bad.
- Interestingly random weight initialization outperforms xavier initialization which is not intuitive. This could be blamed on the reason that the weights are xavier are too small, I checked that the norm of xavier initialized weights is around 5 whereas the norm of weights of radomly initialized weights is around 50, if I bring them to the same order of magnitude then xavier cleraly outperforms random initialization.
- It can also be seen that adaptive gradient optimizers are consistently outperforming SGD, Momentum and NAG.
Clearly, due to their ineffective ability to learn within a finite number of epochs, stochastic gradient descent (SGD), momentum gradient descent, and Nesterov all performed less than optimally due to the poor learning rates of 1e-3/1e-4.
- I found that SGD could reach  more than 85% accuracy with learning_rate set to a much more modest 0.1 on the otherwise same default values that gave the best accuracy.
- After a point the model starts to over fit and so to gain 95% accuracy we can decrease the model complexity and we could also play around with the betas of nadam, adam, rmsprop as through extensive experiments we have shown that these matters the most. I would also suggest to add dropouts and early stopping, switching to convolution neural network will also help, because we can see that the model is able to extract the coarse features but is having problem in detecting finer features, this could be because in the faltten operation we are technically destroying the structure of image and hence the finer details of the image, where as in CNN we dont need to do any of that.  
- Some early runs that had accuracy in the __10%__ region can be attributed to the weight decay set very high, which didnt allow the model to fit the data properly.

## Question 7

Here we plot the confusion matrix, for this we used the __confusion_matrix__ function from __sklearn__ library,
we plot this for the best model found through hyperparameter search in wandb.

![alt text](https://github.com/sasuke-ss1/CS6910/blob/main/Assignment-1/confusion.png)

As we can see that the model is able to predict very well for most classes but as we can see that the model is not properly able to differentiate between coat, dress, pullover and shirt which is resoanable as all of them are clothing, it can differentiate between shoes and clothes very well.

The model also got confused between Shirt and T-shirt, it also confused sneaker and sandals as ankle boots.

Give the above observation, we conclude that our model is able to learn the coarse features from the images, but is not able to fully undersatnd the finer features in the images. 

## Quesiton 8

In this question we compare the __Mean squared error__ with the __Cross entropy__ error emperically, we see that both of them perform quite identically but cross entropy on average is able to perform marginally better that mean squared error, the result is not quite intuitive but could be because MNIST/Fashion-MNIST are very simple dataset and hence cross_entropy and MSE are performing almost the same.

The plots for the same can be found in the wandb report submitted.

## Question 9 and 10

For the most significant 3 hyperparameters that I would like to tune on MNIST given my observations of Fashion-MNIST would be learning_rate, hidden_size and number_of_hidden_layer.

I will choose relu activation as the my best val_accuracy runs were completely dominated by relu, I would also choose nadam as my optimizer as it also performed better for val accuracy accross all the runs.

Batch_size doesnt matter much as we found from the experiments, it also determines the speed of convergence as leeser the batch size, more the training and lesser the epcohs in general, so I will choose batch_size to be 128 also for the experiments it seems that weight initializations also doesnt matter much. Hence I will seacrh for learning_rate, hidden_size, and number_of_hidden_layers.

If we train the model which gave the best result on fashion_mnist on mnist dataset then we see that we are hitting an accuracy of more than __96%__ this hints us that sometime we can simply apply the neural network that worked on a more complex dataset direcly to a less complex dataset.

Here we see that our model performs very vell on MNIST which is less complex than fasion MNIST.

From the consufion matrix plotted for the model that best fitted Fashion_MNIST trained on MNIST we see that the model didnt really get confused much here as learning only the coarse features of the image was sufficient to predict everything accuractely.

![alt text](https://github.com/sasuke-ss1/CS6910/blob/main/Assignment-1/confusion1.png)