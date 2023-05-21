
# Training 

To train this network we run the command 

```sh
python train.py
```
| Name                   | Default Value                    | Description                     |
| :--------------------: | :------------------------------: | ------------------------------- |
| ``` -wp ```, ``` --wandb_project ```  | assignment3   |Project name used to track experiments in Weights & Biases dashboard
| ``` -we ```, ``` --wandb_entity ```   | sasuke        |Wandb Entity used to track experiments in the Weights & Biases dashboard
| ``` -e ```, ``` --epochs ```          | 5             |Number of epochs to train neural network
| ``` -b ```, ``` --batch_size ```      | 32            |Batch size used to train neural network
| ``` -lr ```, ``` --lr ```             | 0.001         |Learning rate used to optimize model parameters
| ``` -es ```, ``` --embedSize ```      | 128            |Size of the embedding vector.
| ``` -hs ```, ``` --hiddenSize ```      | 256            |Dimension of the hidden layer of the backbone.
| ``` -a ```, ``` --attention ```       | False          |Flag for using attention in the decoder.
| ``` -q ```, ``` --question ```        | None          |Question Number of Assignment
| ``` -d ```, ``` --dropout ```         | 0.3           |Dropout probability value for each layer
| ``` -nhl ```, ``` --numHiddenLayers ``` | 2         |SVertical depth of the Encoder/Decoder
| ``` -p ```, ``` --path ```      | ./aksharantar_sampled  |The path to the root directory of the dataset.
| ``` -bi ```, ``` --bidirectional ``` | True            |Flag for training bi-directionaly.
| ``` -l ```, ``` --language  ```    | tam         |The langugae of the dataset.
| ``` -bb ```, ``` --backbone ```    | lstm        |The reccurent model used for encoder and decoder.
| ``` -tf ```, ``` --teacherForcingRatio  ```    | 0.5         |The probabily of using teach forcing training.

Note: More information can be accessed by using ``` python train.py --help ```

## Question 1

# Question 1 (15 Marks)

We build the sequence learning model using ``` pytorch ``` that does transliteration.

A usual Encoder does the following:

- It computes embedding as e = $E_ex$, where x is a $V\times1$ dimensional one-hot vector and $E_e$ is a $m\times V$ dimensional matrix.

- And it computes the hidden state as $h_i =  \sigma (W_ee+U_eh_{i-1}+ b_e)$

Below are the dimensions the same:

<div align="center">

|        | $W_e$ | $U_e$ | $b_e$ |
|--------|-------|-------|-------|
| Dim    | (k,m) | (k,k) | (k,1) |

</div>

A usual decoder does the following:

- It computes its hidden state by the following formula $s_i = \sigma(U_d(e(\hat{y_{i-1}}))+W_ds_{i-1}+b_{d1})$.
- The final outputs of the decoder is then used to compute the softmax probability $P(y) = softmax(V_ds_{i-1}+b_{d2})$.
- We get the one-hot vector representation of the word by doing a max operation across all output possibilities $e(\hat{y}_{i-1}) = E_d \cdot \hat{y_{i-1}}$.

<div align="center">

|        | $W_d$ | $U_d$ | $b_{d1}$ | $V_d$ | $b_{d2}$ | $E_d$ |
|--------|-------|-------|-------|--------|-------|-------|
| Dim    | (k,m) | (k,k) | (k,1) |(V,k) | (V,1) | (m,V) |

</div>

Next, we find the number of computations in the encoder and decoder, for the encoder we have: 

- First we do a matrix multiplication between E and x which takes $Vm$ multiplications and $Vm-m$ additions, so the total computations here $2Vm-m$ which can be written as asymptotically as $O(mV)$.
  
- Similarly, we compute for the hidden state calculation which for $W_ee$ will take $O(mk)$ complexity,$U_eh_{i-1}$ will take $O(k^2)$ complexity, and finally, we will have k additions from $b_e$. Considering exponential to be a constant time operation the sigmoid operation will be of the order $O(k)$. 

Hence the total number of computations are: 

$O(Vm)+O(km)+O(k^2)+O(k) = O(k^2+Km+k+Vm)$

We  do the same analysis for the decoder, where we see:
 
- The multiplication of y and E will again take $O(mV)$ operations. 
- Similarly the computation of $W_ds_{i-1}$ will take $O(mk)$,$U_de(y_{i-1})$ will take $O(k^2)$, and finally there will be k additions due to $b_d$. As mentioned earlier the sigmoid computations will be of the order O(k).
- We see the $V_ds_{i-1}$ will take$O(Vk)$ and V additions, and as the softmax also uses exponential which is a constant time it takes $O(V)$ total operations. 

Hence the total number of computations is:

$O(Vm+mk+k^2)+O(k)+O(Vk)+O(V) = O(Vm+km+k^2+k+V+kV)$

Now all these operations will take place T times so the total number of operations will be:

$O(T(Vm+mk+k^2+k+V+Vk))$


So finally we have $E_e,W_e,U_e,b_e$ are the learnable parameters for the encoder and $W_d,U_d,b_{d1},V_d,b_{d2},E_d$ are the learnable parameters for the decoder. 

So finally the total number of trainable parameters are: $2mV+2mk+2k^2+2k+Vk+V$

## Question 2

We use Bayesian search to optimize the hyperparameter search 

We choose the following hyperparameters to search on:
- backbone: ["rnn", "gru", "lstm"]
- teacher forcing ratio: [0.3, 0.5, 0.7]
- number of hidden layers: [1, 2, 3]
- embedding size: [128, 256, 512]
- dropout: [0.2, 0.3]
- hidden size: [128, 256, 512]
- bidirectional: [True, False]

- We didn't choose many hyperparameters because based on our prior experience we knows what might work the best if tuned properly, we dropped off batch size and epochs from the hyperparameter search as they have no effect on the final best accuracy, batch size only control the number of update per epochs and epochs is just how many sweeps of the data we are completing.
- We also did; not do a hyperparameter search on the optimizer as Adam is known to work the best with the default learning rate of 0.001 and there are a lot more theoretically significant parameters like embedding size, hidden size, etc. to search upon, so we choose here to ignore optimizer and leaning rate in favor of embedding size, hidden size, etc.
-We ran wandb sweep with the target of maximizing the validation accuracy. The intuition behind doing this was that we are training our model for only 5 epochs will give us a general estimate of the validation accuracy the model is capable of. This intuition is further justified by seeing the plots that the curve are flattening out towards the end.

- I wanted to check the impact of varying the teacher forcing ratio on accuracy. Hence, I choose three random by vastly separate values: 0.3, 0.5, and 0.7.

## Question 3

Based on the above plots we conclude the following:
- RNN based model takes converges slower than LSTM and Gru-based models, as can be clearly seen from the graphs, the loss is very high for RNN in comparison to an LSTM and GRU-based model. This is quite intuitive that RNN is performing worse than LSTM and Gru as LSTM and Gru were specifically tailored to combat the shortcomings of RNNs.
- We see that LSTMs have dominated the high accuracy region, which is again intuitive as LSTM was made specifically to work better than RNNs and Gru is sort of a fast approximation to the LSTM which trades off some accuracy for a smaller amount of computations.
- We see that although the models with the best validation accuracies have a high teacher-forcing ratio, they are also the ones that exhibit a high amount of overfitting.(Note: We are setting the teacher-forcing ratio to 0 after 2nd epoch for all the models) The model with a lower teacher-forcing ratio has only marginally lower accuracy but exhibits lower overfitting. The higher teacher forcing does make some sense in a way that initially we are feeding the next correct input to the model and after 2 epochs we are taking that luxury away from the model in the hopes that it now knows how to predict the next character better. 
- We see a positive correlation between hidden size, number of hidden layers and validation accuracy, which is again intuitive as more number of parameters are now available in both the encoder and decoder for them to learn the more complex aspects of the data.
- We also see that there is a negative correlation between embedding size and validation accuracy, which could be explained by the fact that single characters don't have much meaning on their own, and hence span a low dimensional space and projecting it to a higher dimension might result in the model fitting to the noise.
- We see that dropout helped with the overfitting problem and gets a higher accuracy for a higher dropout probability.
- We also see that using a bidirectional sequence model really helped the validation accuracy, which was expected as looking the word from both directions and then predicting the output word is better than just looking at the word from one direction and giving the outputs.

## Question 4
Using the model which worked best on the validation set, we get a test accuracy of __47.99%__.

The whole prediction csv file can be found on the github page named __pred.csv__.


- There are multiple letter in tamil with very close pronounciation, all of which are written the same in English, which confuses the model and hence the error.
Example:
 
<div align="center">

|English   | True       | Predicted |
|----------|------------|-----------|
|paantvitth|பாண்ட்விட்த்|பாண்ட்வித் |
|kallaala  |கல்லால    |கல்லாள    |

</div>

In the first example, tth can be written as "ட்த்" or "த்", hence the confusion. In the second example la can be written as "ல" and aswell "ள". Like hindi, tamil has a few guttural sounds as shown in the above examples, and hence these arent easy to differentiate when written in english and hence stems the confusion. 

- There are some letters in Tamil that are implied in some places and it depends on the writting style, like the hindi letter ka also be written as k.

Example:

<div align="center">

|English   | True       | Predicted |
|----------|------------|-----------|
|kidanta   |கிடந்த      |கிடண்ட    |
|tokaiyin  |தொகையின் |டோகையின்|

</div>

In both examples we see a common pattern that "t" can be represented as "த" or "ட" with the different matras.

- The model added repreated characters at someplaces.

Example:

<div align="center">

|English   | True       | Predicted |
|-----------|------------|-----------|
|syriyap     |சிரியப்     |சிற்யபப   |
|jallian    |ஜாலியன்    |ஜலலலியான்|

</div>

## Question 5

We add the attention mechanism with the following lines of code over a generic decoder:

``` python 
        # Calculating Attention
        batchSize, seqLen, _ = encoderOutputs.shape
        hidden = hidden.unsqueeze(1).repeat(1, seqLen, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoderOutputs), dim = 2)))
        
        attention = self.v(energy).squeeze(2)
  
        attnWeights = F.softmax(attention, dim=1).unsqueeze(1)

        # Apply Attention
        attnApplied = torch.bmm(attnWeights, encoderOutputs).transpose(0, 1)
        input = torch.cat((embed, attnApplied), dim = 2).permute(1,0,2)

        output, hidden = self.seq(input, s)

        output = self.softmax(F.relu(self.out(output)))
```

Add an attention network to your base sequence-to-sequence model and train the model again. For the sake of simplicity, you can use a single-layered encoder and a single-layered decoder (if you want, you can use multiple layers also). Please answer the following questions:

We got the test accuracy to be __41.35%__, The predictions could be found on the github page in the file named __predAttn__.
 
- In our setting of transliteration, the attention mechanism will have limited uses when compared to neural machine translation or question-answering tasks, as in characters have limited correlation when compared to sentences and passages and attention is harder to train as the weights are trained in an unsupervised manner, so it takes more number of epochs to trains with learning rate scheduling, etc. which explains a lower test set accuracy than vanilla seq2seq model.

- That being said we still see that the accuracies for the initial epochs have improved by quite a lot when compared to the vanilla seq2seq, we also have rnn's now giving decent validation word accuracy of about 40%, we also see that attention also far surpasses the vanilla model in terms of average model accuracies as in the worst performers for attention are still better than vanilla seq2seq. 

- On average, the accuracy of all the models did improve somewhat but the best model is still not getting an accuracy of 50% what did improve is that now there are very less models below 25% accuracy.We also see that attention doesn't work for multilayered encoders and decoders, it also gives slightly worse results for bidirectional encoders and decoders. If we compare the exact configurations of the models for attention and non attention, we find that for single layered encoder and decoders the attention improves the accuracy by a lot.


The model corrects the following words:

<div align="center">

|English   | True       | Predicted | Predicted(Attention) |
|-----------|------------|-----------|-------------------- |
|praem      |பிரேம்       |ப்ரே      | பிரேம்              |
|mutiyala   |முடிய       |முட்டியால| முடியல             |
|kallaala   |கல்லால     |கல்லால   | கல்லாள             | 

</div>

There are a few more examples, the attention model also fixes the charecter level predictions for many words, a few are shown below:

<div align="center">

|English   | True       | Predicted | Predicted(Attention) |
|-----------|------------|-----------|-------------------- |
|grover     |க்ரோவர்    |குரோவ்ர  |  கிரோவரர்           |
|kidanta    |கிடந்த      | கிடண்ட  | கிடன்டா             |
|under      |அண்டர் |யு்டரர | உண்டர்   | 
|kollaadhavar |கொள்ளாதவர் |கொல்லாதவர் | கொள்லாதவர்| 
|sarbia |செர்பியா | சர்்ியா |  சர்பியா   | 
|loova |லோவ |லூவா | லோவா | 

</div>

Below are the and thea heat maps:

![alt text](https://github.com/sasuke-ss1/CS6910/blob/main/Assignment-1/attnMap.png)

Since Python lacks built-in support for the Tamil font, we only plot the English names. We see that while printing outputs for the PAD, SOW, and EOW token the attention scores are uniform the model isn't giving any weightage to the actual English word but the model is giving weights to the other pad tokens, we also see that the mapping between them is predominantly diagonal implying that Tamil letters align alternatively with the English letters in a sequential way. We also see that a single Tamil letter gives attention to multiple rows suggesting the presence of syllables in English words. Hence proving that our attention mechanism is working.
