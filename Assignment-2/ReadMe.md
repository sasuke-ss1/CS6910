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

The images in the INaturalist dataset have variable sizes and can be grayscale as well as rgb, so we make a __NatureData__ class that loads the images applies all the data augementation that we supply as arguments to the class, it can also load test data by setting ``` train = False ```.

To ensure that the image matches the model input size we resize the images to 224x224(Details in question-2).

As for our main model, we first create a block that does convolution, activation, maxpool, dropout and then batch normalization. We use this block to build our five layer network, we then put a ``` Flatten() ``` layer and after that we add a ``` Dense() ``` layer with appropriate input dimention and output dimention, we dont apply any activation to the output of the last layer.

Every convulutional layer can have variable filter size and num of filters, we did fix the stride to be 1 for convolutional layers and maxpool kernel size to be (2,2), the stride of convolutional layers are fixed to be 1 as, stride si meant to shrink the size of the feature maps and in this case maxpool is already doing that.

The last layers contains 10 output neurons for the 10 classes, and the input to the last layer is claculated by using the below formula and assuming the feature maps to be square all the time.

$$
H_{out} = \lfloor \frac{H_{in} + 2*padding - dilation*(kernel\_size -1) -1}{stride} + 1
$$

This is the output dimension of a convolutional layer as well as maxpool layer,  just that in the case of maxpool layer stride is 2 and in convolutional layer the stride is 1.

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

## Question 2

To split the training data into 80-20% train-val split we use the inbuilt pytorch function ``` random_split() ``` this as the name suggests splits the data randomly into two, with the specified configuration.

We also apply the various inbuilts "transforms" to augment our data as already our data is quite less when compared to the conventional datasets like CIFAR10 which have 60,000 instances.

We dont apply the augmentation to the test data.

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

We then write a train loop for the wandb sweep function as initialize the sweep values to the vlues given below.

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

Like the last assingment we use bayesian search over the hyper parameteres, because it actually considers the past history of hyperparameters(prior) selected before selecting the new set of hyperparameters.

So at every run it does the following:
- Choose a set of hyperparameter values (our belief) and use them to train the machine learning model.
- Evaluate the performance of the model on those hyperparameters.
- Update our belief accordingly.

So to summarize we begin with an initial estimate of the hyperparameters and progressively update it in light of previous findings.

We choose only those hyperparameters that theoriticially makes sense instead of doing an extensive search over everything:
- We choose learning rate to be either 0.001 or 0.0001 as we were using Adam optimizer and these two learning rate have been emperically shown to work the best.
- We only use variants of ReLU as activation as again these are know to work best for any neural network configuration, we also have the dropout and batch_norm in our hyperparameters as they theoritically help with overfitting.
- The other hyperparamters are just selection over some model configurations like filter size and number of filters per layer which affect the number of parameters in our model.

This foresight along with the bayesian stratergy helped us to get the best accuracies in only 30 runs.

Below are the plots for the same.

##  Question 3 

Our best model used learning rate 0.0001 and 64 filters with filter sizes 5 at the first two convolution layers and 3 for the rest of the layers.

- We see that it used dropout to be 0.3 and it __didnt__ use batch normalization. This can be explained by the fact that we didnt actually train it more more number of epochs which could have lead to over fitting, batch normalization prevents the network from overfitting and hence results in a stable training, hence we didnt require batch normalization here.
- We also see that having larger filter size at the starting layers and smaller filters at the deeper layers was favoured during training instead of all smaller filters and all larger filters, this can be explained qualitatively by thinking that first the network need to abstarct the general portions of the image for which having a larger receptive field(filter size) is favourable, and later the filter need to learn the specifics and tiny details of the image which the smaller receptive field(filter size) is best.
- We also see that 0.3 dropout probability is favoured over 0.5, this can be explained by the fact that the model was only allowed to train for 10 epochs and in the case of $p=0.5$ at every epoch the model was working only at half the capacity so it couldnt properly fit the data, where as 0.3 gave a perfect balance between train time and accuracy.
- We see that the search selected the most complex possible to be the best model, this model is prone to overfitting as it contains many parameters, but if given enough data this model will outperform any model with lesser complexity, this model did the best because it had many parameters and hence led to faster training.
- We also see that 0.0001 is selected as the learning rate and LeakyReLU as the activation function, there is not much to say about these choices as it depends to the dataset and the optimization problem at hand, it is always best to do a hyperparameter search to get the best values.
- We see that the most important hyperparameters were those that affected the model complexity as expected.

## Question 4&5

We get the best accuracy on the test set to be __43.4%__.

Shown below are the 10 predictions of the model for 3 classes with the confidence of the model on each prediction also mentioned.

# Part B

## Question 1

We are using __ResNet50__ for our finetune/tranfer learning needs, we notice that it was pretrained on images of shape 224x224 and hence will need images of the shame shape as inputs. Our images have varaible height and width but all are greater than 224, so we can simply resize it (224, 224) without much loss of information, or we can also do random cropping to augment and rescale oru images at the same time, both of these methods results in some loss of information.

If we are rigid on passing high quality images(512, 512) with minimum rescalling, we can have a CNN before the input of the ResNet50 that takes 512x512 images as inputs and outputs 224x224 feature maps. Intution here is that the CNNs wil learn the proper "rescaling" for each image and then supply the images to the actuall model. To properly train this we could first fix the entire ResNet and only train the "rescaling" CNN for a few epochs and then fine tune the entire network.

As for the output of linear layer, there are 1000 classes in IMAGENET dataset and we only need 10 output nodes as we have only 10 classes to predict. This can be done by changing the final linear layers output to 10 instead of 1000 and then train only the last layer while keeping the rest of the network fixed, or we can do the same thing as above and first train only the linear layer for a few epochs and then train the enitre network.

We can also add another linear layer on top of the linear layer already present and then train the network according to the aformentioned stratergies.

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

## Question 2

In general the models used to finetuning/transfer learning are very big and take a lot of time and gpu usage to do any sensible amount of training. We propose some stratergies to finetune the model in a tractable manner:

- We could first make all the parameters of the model untrainable and then add/replace the last linear layer to match our prediction specifications and only keep this last layer as trainable and start the training. This method really helps if you have a scarse amount of data as if you were to train the entire network to this data the model can severly overfit due to the massive number of paramters these model have.
- If we have a bigger dataset we can unfreeze a few more last layers and train the model, one more thing that intuitively makes sense if that we should keep the learning rate for this smaller and smaller as we unfreeze more layers for training, as the model has converged for IMAGENET and our dataset which has classes similar to that of IMAGENET needs to only slighly adjust the weights to get the best results. This argumnet relies on the fact that the model was trained using a large number of data points and hence is already in a good local minima.
- General heuristic is that model learn more general features that are useful for a wide range of tasks, while higher layers learn more task-specific features so one can do selective finetuneing. By selectively fine-tuning specific layers of the pre-trained model, we can quickly adapt the model to the new task, while minimizing the risk of overfitting or forgetting the knowledge learned during pre-training

## Question 3 

We are using the first of the aforementioned stratergy of only training the final Linear layer and use the ResNet backbone as a feature extractor:

We observe the following:
- We ran a wandb sweep over three different values of learning rate and we see that for all the different values of learning rate we are getting test accuracy of over __70%__ which is a significantly high when compared to the best "trained from scratch" accuracy which was only __43.4%__, this just goes to show the validtity of point 1 in the previous question, we didnt have sufficient data to learn a powerfull classifier from scratch, whereas finetuning the last layer required minimal data to give very goo d accuracies.
- We see that the best finetune model was able to get __79-80%__ accuracy in only 10 epochs which is again a 4 time improvement from the scratch model which took 40 epochs to train. This result is also quite intuitive as the finetune model only needs to train a Linear layer where as for the scratch model we need to train the entire model which takes a lot of time.
- We also see that 0.00001 learning rate performed the best among all possible choice this support point 2 in the above question as we used 0.0001 learning rate for scratch training.

Below are the plots for the same: