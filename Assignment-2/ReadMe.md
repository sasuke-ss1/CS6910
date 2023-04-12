# Part A

## Training

To train this network we run the command 

```sh
python trainA.py
```
| Name                   | Default Value                    | Description                     |
| :--------------------: | :------------------------------: | ------------------------------- |
| ``` -wp ```, ``` --wandb_project ```  | assignment2   |Project name used to track experiments in Weights & Biases dashboard
| ``` -we ```, ``` --wandb_entity ```   | sasuke        |Wandb Entity used to track experiments in the Weights & Biases dashboard
| ``` -e ```, ``` --epochs ```          | 10            |Number of epochs to train neural network
| ``` -b ```, ``` --batch_size ```      | 64            |Batch size used to train neural network
| ``` -lr ```, ``` --lr ```             | 0.001         |Learning rate used to optimize model parameters
| ``` -nf ```, ``` --num_filters ```    | 32            |Number of hidden neurons in a feedforward layer
| ``` -a ```, ``` --activation ```      | ReLU          |Activation function after each layer
| ``` -q ```, ``` --question ```        | None          |Question Number of Assignment
| ``` -d ```, ``` --dropout ```         | 0.3           |Dropout probability value for each layer
| ``` -bn ```, ``` --bacth_norm ```     | False         |Set true to apply batch normalization to every layer.
| ``` -p ```, ``` --parent_dir ```      | ./nature_12K  |Path to the parent directory of the dataset.
| ``` -fs ```, ``` --filter_size ```    | 3             |Filter size of each layer seperated by comma
| ``` -fo ```, ``` --filter_org  ```    | const         |Stratergy for depth of each layer's activation

Note: More information can be accessed by using ``` python trainA.py --help ```

__Please note the directory structure should be as shown below__

Directory structure:
```
    ├── ...
    ├── parent_dir
    │   ├── train
    │   │   └── class_folders
    │   │       └── *.png
    │   │   
    │   ├── val    
    │   │   └── class_folders     
    │   │       └── *.png       
    │   │                  
    └── ...
```

## Question 1

The images in the INaturalist dataset have variable sizes and can be grayscale as well as RGB, so we make a __NatureData__ class that loads the images and applies all the data augmentation that we supply as arguments to the class, it can also load test data by setting ``` train = False ```.

To ensure that the image matches the model input size we resize the images to 224x224(Details in question-2).

As for our main model, we first create a block that does convolution, activation, max pool, dropout, and then batch normalization. We use this block to build our five-layer network, we then put a ``` Flatten() ``` layer, and after that, we add a ``` Dense() ``` layer with appropriate input dimension and output dimension, we don't apply any activation to the output of the last layer.

Every convolutional layer can have variable filter size and num of filters, we did fix the stride to be 1 for convolutional layers and max pool kernel size to be (2,2), the stride of convolutional layers is fixed to be 1 as stride si meant to shrink the size of the feature maps and in this case, max pool is already doing that.

The last layers contain 10 output neurons for the 10 classes, and the input to the last layer is calculated by using the below formula and assuming the feature maps to be square all the time.

$$
H_{out} = \lfloor \frac{H_{in} + 2*padding - dilation*(kernel\_size -1) -1}{stride} + 1 \rfloor
$$

This is the output dimension of a convolutional layer as well as the max pool layer,  just that in the case of the max pool layer stride is 2 and in the convolutional layer the stride is 1.

Below is the code for the model class:

``` python 
class Model(nn.Module):
    def __init__(self, num_filters, filter_size, pool_size, activation, img_shape=224, dropout=None, batch_norm=False, num_classes=10):
        super().__init__()
        if len(filter_size) == 1:
            filter_size = 5*[filter_size[0]]
        layers = [];dim = img_shape
        for i in range(5):
            layers.append(convActPool(num_filters[i], num_filters[i+1], filter_size[i], pool_size, activation))
            dim = dim + 2-(filter_size[i]-1)
            dim = (dim - (2-1)-1)//2 + 1
        
        layers.append(nn.Flatten())    
        layers.append(nn.Linear(dim*dim*num_filters[-1], 10))
        
        self.logits = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.logits(x)
        return logits


class convActPool(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, pool_size, activation, dropout=None, batch_norm=False):
        super().__init__()
        conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=1)
        activation = getattr(nn, activation)()
        pool = nn.MaxPool2d(pool_size)
        layers = [conv, activation, pool]

        if dropout:
            layers.append(nn.Dropout(dropout))
        
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channel))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out = self.network(x)
        
        return out

```

Below is the code for the dataset class:

``` python
class NatureData(Dataset):
    def __init__(self, dir, train=True, transforms=None):
        super().__init__()
        self.train_dir = os.path.join(dir, "train")
        self.test_dir = os.path.join(dir ,"val")
        self.transforms = transforms

        self.classes = list(map(lambda x: x.split("/")[-1], glob(self.train_dir + "/*")))
        self.idx = list(range(len(self.classes)))
        self.train = train
        self.cltoidx = dict(zip(self.classes, self.idx))

        self.train_img_path, self.test_img_path = [], []
        for cl in self.classes:
            self.train_img_path += glob(os.path.join(self.train_dir, cl) + "/*")
            self.test_img_path += glob(os.path.join(self.test_dir, cl) + "/*")

    def __len__(self):
        if self.train:
            return len(self.train_img_path)

        return len(self.test_img_path)

    def __getitem__(self, idx):
        if self.train:
            path = self.train_img_path[idx]
        else:
            path = self.test_img_path[idx]
        index = self.cltoidx[path.split("/")[-2]]
        img = Image.open(path).convert('RGB')
        label = index

        if self.transforms:
            img = self.transforms(img)

            return img, label

```

We define the number of computations done by the network as the number of multiplication and additions and the number of parameters is defined as the number of __numbers__ that the network has to decide itself by using the data as a cue.


Below is a table that has the number of parameters and the number of operations performed by each layer:

- The input is assumed to be of the size (H, W, 3) i.e. a general image, we used H = W = 224.
- As mentioned above the output for this input is given by:
$$
H_{out} = \lfloor \frac{H_{in} + 2*padding - dilation*(kernel\_size -1) -1}{stride} + 1 \rfloor
$$
$$
W_{out} = \lfloor \frac{W_{in} + 2*padding - dilation*(kernel\_size -1) -1}{stride} + 1 \rfloor
$$

- The filter sizes of the convolution layer is assumed to be kxk and there are m filters in each layer.


| layers                 | No. of parameters                | No. of operations                                      | 
| :--------------------: | :------------------------------: | ------------------------------------------------------ |
| Conv1                  | $k*k*m*3$                        | $(H−(k−1))∗(W−(k−1))∗m*3*k^2+(H−(k−1))∗(W−(k−1))∗m$     
| MaxPool1               | $0$                              | $(H−(k−1))∗(W−(k−1))∗m$                                
| Conv2                  | $k*k*m*m$                        | $(H−3(k−1))∗(W−3(k−1))∗m*m*k^2+(H−3(k−1))∗(W−3(k−1))∗m$ 
| MaxPool2               | $0$                              | $(H−3(k−1))∗(W−3(k−1))∗m$                              
| Conv3                  | $k*k*m*m$                        | $(H−5(k−1))∗(W−5(k−1))∗m*m*k^2+(H−5(k−1))∗(W−5(k−1))∗m$
| MaxPool3               | $0$                              | $(H−5(k−1))∗(W−5(k−1))∗m$                              
| Conv4                  | $k*k*m*m$                        | $(H−7(k−1))∗(W−7(k−1))∗m*m*k^2+(H−7(k−1))∗(W−7(k−1))∗m$
| MaxPool4               | $0$                              | $(H−7(k−1))∗(W−7(k−1))∗m$                              
| Conv5                  | $k*k*m*m$                        | $(H−9(k−1))∗(W−9(k−1))∗m*m*k^2+(H−9(k−1))∗(W−9(k−1))∗m$
| MaxPool5               | $0$                              | $(H−9(k−1))∗(W−9(k−1))∗m$    
                    

The total number of parameters and the total number of operations is the column-wise sum of __No. of parameters__ and __No. of operations__ respectively.  

## Question 2

To split the training data into 80-20% train-val split we use the inbuilt pytorch function ``` random_split() ``` this as the name suggests splits the data randomly into two, with the specified configuration.

We also apply the various inbuilt "transforms" to augment our data as already our data is quite less when compared to the conventional datasets like CIFAR10 which have 60,000 instances.

We don't apply augmentation to the test data.

Below is the code for all the transformations applied:

``` python 

train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation((180)),
            transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter()]), p=0.5),
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomErasing(p=0.4, value='random')

])

val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

```

We then write a train loop for the wandb sweep function as initializing the sweep values to the values given below.

```
program: train.py
method: bayes
name: "complete-sweep"
metric:
  name: val_accuracy
  goal: maximize
parameters:
    lr:
      values: [0.001, 0.0001]
    activation:
      values: ["SiLU", "LeakyReLU", "ReLU", "GELU"]
    batch_norm:
      values: [True, False]
    dropout:
      values: [0.3, 0.5]
    filter_org: 
      values: ["half", "double", "const"]
    filter_size:
      values: ["3", "5,5,3,3,3", "5"]
    num_filters:
      values: [16, 32, 64]

```

Like the last assignment, we use bayesian search over the hyperparameters because it actually considers the past history of hyperparameters(prior) selected before selecting the new set of hyperparameters.

So at every run, it does the following:
- Choose a set of hyperparameter values (our belief) and use them for training the machine learning model.
- Evaluate the performance of the model on those hyperparameters.
- Update our belief accordingly.

So to summarize we begin with an initial estimate of the hyperparameters and progressively update it in light of previous findings.

We choose only those hyperparameters that theoretically make sense instead of doing an extensive search over everything:
- We chose the learning rate to be either 0.001 or 0.0001 as we were using Adam optimizer and these two learning rates have been empirically shown to work the best.
- We only use variants of ReLU as activation as again these are known to work best for any neural network configuration, we also have the dropout and batch_norm in our hyperparameters as they theoretically help with overfitting.
- The other hyperparameters are just selections over some model configurations like filter size and number of filters per layer which affect the number of parameters in our model.

This foresight along with the bayesian strategy helped us to get the best accuracies in only 30 runs.

We also ran wandb sweep with the target of actually maximizing the train accuracy and we have also plotted the graphs for that aswell.

The intuition behind doing this was that we are training our model for only 10 epochs so there should be minimal overfitting so technically we should get about the same parameter configurations across both the sweeps.

##  Question 3 

Our best model that maximized the validation accuracy used a learning rate of 0.001, GELU activation, no batch norm, dropout to be 0.5, filter org constant and filter size of 3.
- We see that the hyperparameter search resulted in the selection of a model that has a very low number of parameters,  we see that the 2nd best run has the number of filters in each layer as 16 which has an even less number of parameters but it had a lower accuracy, it might be because it has too less number of parameters to fit well to the data.
- We see that we used a filter size of 3 across all the layers and most of the good accuracy models used the same.
- The dropout chosen by the hyperparameter search is 0.5 which is very high as the model is working at half capacity all the time,  the search also selected the learning rate of 0.001 and not 0.0001, this can be explained by seeing that our model is already working at half capacity so it makes sense to have a higher learning rate so every parameter gets reasonably updated.
- We see that batch norm was discarded for the top selection but was considered for the 2nd and 3rd ranked model, batch norm might not be working here because we were using a mini batch_size of 32 and this will not give the best representation of the distribution of the data.
- We see that we have a lot of regularization factors in the best model selected, although this model has a very less chance of overfitting it will take a lot of time to train to the same accuracies an unregularized and complex model can achieve in only a few epochs.

Our best model that maximized the training accuracy used a learning rate of 0.0001 and 64 filters with filter sizes 5 for the first two convolution layers and 3 for the rest of the layers.

- We see that it used dropout to be 0.3 and it __didn't__ use batch normalization. This can be explained by the fact that we didn't actually train it more number epochs which could have led to overfitting, batch normalization prevents the network from overfitting and hence results in stable training, hence we didn't require batch normalization here, also as mentioned above it also might be because that our mini-batch statistics are just not good enough for the batch norm to work.
- We also see that having larger filter size at the starting layers and smaller filters at the deeper layers were favoured during training instead of all smaller filters and all larger filters, this can be explained qualitatively by thinking that first, the network needs to abstract the general portions of the image for which having a larger receptive field(filter size) is favourable, and later the filter need to learn the specifics and tiny details of the image which the smaller receptive field(filter size) is best for training.
- We also see that 0.3 dropout probability is favoured over 0.5, this can be explained by the fact that the model was only allowed to train for 10 epochs and in the case of $p=0.5$ at every epoch the model was working only at half the capacity so it couldn't properly fit the data, whereas 0.3 gave a perfect balance between train time and accuracy.
- We see that the search selected the most complex possible to be the best model which as expected, this model is prone to overfitting as it contains many parameters, but if given enough data this model will outperform any model with lesser complexity, this model did the best because it had many parameters and hence led to faster training.
- We also see that 0.0001 is selected as the learning rate and LeakyReLU as the activation function, there is not much to say about these choices as it depends on the dataset and the optimization problem at hand, it is always best to do a hyperparameter search to get the best values.
- We see that the most important hyperparameters were those that affected the model complexity as expected.

Joint Conclusions:
- We see a significant difference in the hyperparameters selected for maximum validation and training accuracy. We see that for higher validation accuracies the search results in highly regularized and smaller models get the upper hand whereas for maximum training accuracies, we see that the models that have higher complexities and fewer regularization wins, which was completely expected, but that being said the actual accuracy difference between these two is minimal.
- As hinted earlier, highly regularized models with lower model complexities won't overfit but it will take a lot of training time to actually get some decent accuracies and after some time the validation and training accuracy will reach a maximum(not necessarily high) because of the model complexity.
The same thing is true for complex models as well just that after a point the validation peaks and the training accuracy only grows(basically overfitting) but this time the training is significantly faster.
- We confirm the above point by training the max. validation accuracy model for 50 epochs and training the max. training accuracy model for only 30 epochs and we see that we get __43.4%__ accuracy on the test set for the latter and __43.25%__ accuracy for the former.

## Question 4&5

We get the best accuracy on the test set to be __43.4%__.

Shown below are the 10 predictions of the model for 3 classes with the confidence of the model on each prediction also mentioned.

![alt text](https://github.com/sasuke-ss1/CS6910/blob/main/Assignment-2/1.png)

# Part B

## Training
To train this network we run the command 

```sh
python trainB.py
```
| Name                   | Default Value                    | Description                     |
| :--------------------: | :------------------------------: | ------------------------------- |
| ``` -wp ```, ``` --wandb_project ```  | assignment2   |Project name used to track experiments in Weights & Biases dashboard
| ``` -we ```, ``` --wandb_entity ```   | sasuke        |Wandb Entity used to track experiments in the Weights & Biases dashboard
| ``` -e ```, ``` --epochs ```          | 10            |Number of epochs to train neural network
| ``` -b ```, ``` --batch_size ```      | 64            |Batch size used to train neural network
| ``` -lr ```, ``` --lr ```             | 0.001         |Learning rate used to optimize model parameters
| ``` -q ```, ``` --question ```        | None          |Set True to run wandb experiments
| ``` -p ```, ``` --parent_dir ```      | ./nature_12K  |Path to the parent directory of the dataset.

Note: More information can be accessed by using ``` python trainB.py --help ```

__Please note the directory structure should be as shown below__

Directory structure:
```
    ├── ...
    ├── parent_dir
    │   ├── train
    │   │   └── class_folders
    │   │       └── *.png
    │   │   
    │   ├── val    
    │   │   └── class_folders     
    │   │       └── *.png       
    │   │                  
    └── ...
```

## Question 1

We are using __ResNet50__ for our finetune/transfer learning needs, we notice that it was pretrained on images of shape 224x224 and hence will need images of the same shape as inputs. Our images have variable heights and widths but all are greater than 224, so we can simply resize them (224, 224) without much loss of information, or we can also do random cropping to augment and rescale our images at the same time, both of these methods results in some loss of information.

If we are rigid on passing high-quality images(512, 512) with minimum rescaling, we can have a CNN before the input of the ResNet50 that takes 512x512 images as inputs and outputs 224x224 feature maps. The intuition here is that the CNNs will learn the proper "rescaling" for each image and then supply the images to the actual model. To properly train this we could first fix the entire ResNet and only train the "rescaling" CNN for a few epochs and then fine-tune the entire network.

As for the output of the linear layer, there are 1000 classes in the IMAGENET dataset and we only need 10 output nodes as we have only 10 classes to predict. This can be done by changing the final linear layers output to 10 instead of 1000 and then training only the last layer while keeping the rest of the network fixed, or we can do the same thing as the above and first train only the linear layer for a few epochs and then train the entire network.

We can also add another linear layer on top of the linear layer already present and then train the network according to the aforementioned strategies.

Code for the same:

```python

def get_resnet(num_classes, fineTune = True):
    ResNet = resnet50(pretrained=True)
    if fineTune:
        for param in ResNet.parameters():
            param.requires_grad = False

    n_inputs = ResNet.fc.in_features

    ResNet.fc = nn.Sequential(
        nn.Linear(n_inputs,2048),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(2048, num_classes)
        )
                
        #return ResNet
    
    return ResNet

```


```python

def get_resnet(num_classes, fineTune = True):
    ResNet = resnet50(pretrained=True)
    if fineTune:
        for param in ResNet.parameters():
            param.requires_grad = False

    n_inputs = ResNet.fc.in_features

    ResNet.fc = nn.Sequential(
        nn.Linear(n_inputs,2048),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(2048, num_classes)
        )
                
        #return ResNet
    
    return ResNet

```

## Question 2

In general, the models used for finetuning/transfer learning are very big and take a lot of time and GPU usage to do any sensible amount of training. We propose some strategies to finetune the model in a tractable manner:

- We could first make all the parameters of the model untrainable and then add/replace the last linear layer to match our prediction specifications and only keep this last layer as trainable and start the training. This method really helps if you have a scarce amount of data as if you were to train the entire network to this data the model can severely overfit due to the massive number of parameters these models have.
- If we have a bigger dataset we can unfreeze a few more last layers and train the model, one more thing that intuitively makes sense is that we should keep the learning rate for this smaller and smaller as we unfreeze more layers for training, as the model has converged for IMAGENET and our dataset which has classes similar to that of IMAGENET needs to only slightly adjust the weights to get the best results. This argument relies on the fact that the model was trained using a large number of data points and hence is already in a good local minimum.
- General heuristic is that model learns more general features that are useful for a wide range of tasks, while higher layers learn more task-specific features so one can do selective finetuning. By selectively fine-tuning specific layers of the pre-trained model, we can quickly adapt the model to the new task, while minimizing the risk of overfitting or forgetting the knowledge learned during pre-training

## Question 3&4

We are using the first of the aforementioned strategy of only training the final Linear layer and using the ResNet backbone as a feature extractor:

We observe the following:
- We ran a wandb sweep over three different values of learning rate and we see that for all the different values of learning rate we are getting test accuracy of over __75%__ which is significantly high when compared to the best "trained from scratch" accuracy which was only __43.4%__, this just goes to show the validity of point 1 in the previous question, we didn't have sufficient data to learn a powerful classifier from scratch, whereas finetuning the last layer required minimal data to give very good accuracies.
- We see that the best finetune model was able to get __79-80%__ accuracy in only 6 epochs which is again a 4-time improvement from the scratch model which took 40 epochs to train. This result is also quite intuitive as the finetune model only needs to train a Linear layer whereas, for the scratch model, we need to train the entire model which takes a lot of time.
- We also see that the 0.00001 learning rate performed the best among all possible choice this support point 2 in the above question as we used a 0.0001 learning rate for scratch training. The best test accuracy was __81.8%__.

