from dataset import dataset
from Model import *
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.optim import Adam
import random
from tqdm import tqdm
import sys
from argparse import ArgumentParser
from utils import *
import numpy as np
import wandb
import yaml

torch.manual_seed(10)
np.random.seed(10)

parser = ArgumentParser()
parser.add_argument("--wandb_project", "-wp", default="test-2", type=str, help="Project name used to track experiments in Weights & Biases dashboard")
parser.add_argument("--wandb_entity", "-we", default="sasuke", type=str, help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
parser.add_argument("--path", "-p", type=str, default="aksharantar_sampled", help="The path to the root directory of the dataset.")
parser.add_argument("--learningRate", "-lr", type=float, default=1e-3, help="Learning rate to train the model")
parser.add_argument("--backbone", "-bb", type=str, default="gru", help="The reccurent model used for encoder and decoder.")
parser.add_argument("--hiddenSize", "-hs", type=int, default=512, help="Dimension of the hidden layer of the backbone.")
parser.add_argument("--embedSize", "-es", type=int, default=256, help="Dimension of the embeddings.")
parser.add_argument("--epochs", "-e", default=5, type=int, help="Number of epochs to train the model")
parser.add_argument("--batch_size", "-bs", default=32, type=int, help="Batch Size for the data.")
parser.add_argument("--teacherForcingRatio", "-tf", type=float, default=0.0, help="The probabily of using teach forcing training.")
parser.add_argument("--attention", "-a", type=lambda x: (str(x).lower() == 'true'), default=False, help="Flag for using attention in the decoder.")
parser.add_argument("--numHiddenLayers", "-nhl", type=int, default=2, help="Vertical depth of the Encoder")
parser.add_argument("--bidirectional", "-bi", type=lambda x: (str(x).lower() == 'true'), default=False, help="Flag for training bi-directional.")
parser.add_argument("--dropout", "-d", type=float, default=0.25, help="Probability of dropout.")
parser.add_argument("--language", "-l", type=str, default="tam", help="The langugae of the dataset.")
parser.add_argument("--question", "-q", required=False, type=int,help="The wandb question number to run.")
#parser.add_argument("--beamS")

debug = False

args = parser.parse_args()
path = args.path
teacherForcingRatio = args.teacherForcingRatio


trainData = dataset(path, lang=args.language, typ="train")
valData = dataset(path, lang=args.language, typ="val")
testData = dataset(path, lang=args.language, typ="test")

trainLoader = DataLoader(trainData, batch_size=args.batch_size)
valLoader = DataLoader(valData, batch_size=args.batch_size)
testLoader = DataLoader(testData, batch_size=args.batch_size)

sowToken = trainData.x2TDict["\t"] #hardcoded
eowToken = trainData.x2TDict["\n"] #hardcoded

device = torch.device("cuda" if torch.cuda.is_available() and not debug else "cpu")

print(f"Training is happening on: {device}")


inputSize = trainData.xLen
outputSize = trainData.yLen

def train(inputTensor: torch.Tensor, targetTensor: torch.Tensor,\
            encoder: Encoder, decoder: Decoder, \
            encOptim: torch.optim, decOptim: torch.optim, \
            criterion: nn.Module, maxLen: int, useAttn: bool, teacherForcingRatio=teacherForcingRatio):

    encHidden = encoder.initHidden(args.batch_size, device)

    encOptim.zero_grad()
    decOptim.zero_grad()

    inputLen = inputTensor.size(1)
    targetLen = targetTensor.size(1)
 

    loss = 0
    
    encOutput, encHidden = encoder(inputTensor, encHidden)
    
    decHidden = encHidden
    decInput = targetTensor[:, 0]

    useTeacherForcing = True if random.random() < teacherForcingRatio else False

    outputs = torch.zeros(args.batch_size, targetLen, trainData.yLen).to(device)

    if useTeacherForcing:
        # Teacher forcing: Feed the target as the next input

        # Check for attention:
        if useAttn:
            for di in range(1, targetLen):
                decOutput, decHidden, decoderAttention = decoder(decInput, decHidden, encOutput)
                
                outputs[:, di] = decOutput.squeeze(1)

                decInput = targetTensor[:, di]  # Teacher forcing
        else:   
            for di in range(1, targetLen):
                decOutput, decHidden = decoder(decInput, decHidden)
                
                outputs[:, di] = decOutput.squeeze(1)
                
                decInput = targetTensor[:, di]  # Teacher forcing           

    else:
        # Without teacher forcing: use its own predictions as the next input
        if useAttn:
            for di in range(targetLen):
                decOutput, decHidden, decAttn = decoder(decInput, decHidden, encOutput)

                outputs[:, di] = decOutput.squeeze(1)
                
                top= decOutput.argmax(-1)
                decInput = top.squeeze(1).detach()  # detach from history as input
                # EOW   

        else:
            for di in range(targetLen):
                decOutput, decHidden = decoder(decInput, decHidden)

                outputs[:, di] = decOutput.squeeze(1)
                
                top= decOutput.argmax(-1)
                decInput = top.squeeze(1).detach()  # detach from history as input
                # EOW   

    outputs = outputs.permute(0,2,1)

    loss += criterion(outputs, targetTensor) 

    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1)
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1)

    encOptim.step()
    decOptim.step()

    AccW = wordAccuracy(outputs, targetTensor)
    AccC = charAccuracy(outputs, targetTensor)

    return loss.item(), AccW, AccC


def evaluate(enc: Encoder, dec: Decoder, pair: list, criterion: nn.Module, useAttn:bool, ret=False):
    with torch.no_grad():
        inputTensor = pair[0].to(device)
        targetTensor = pair[1].to(device)

        inputLen = inputTensor.size(1)
        targetLen = targetTensor.size(1)
        
        encHidden = enc.initHidden(args.batch_size, device)
        Attn = torch.zeros((targetLen, args.batch_size, inputTensor.shape[-1]))

        outputs = torch.zeros(args.batch_size, targetLen, trainData.yLen).to(device)

        loss = 0
        encOutput, encHidden = enc(inputTensor, encHidden)


        decHidden = encHidden
        decInput = targetTensor[:, 0]

        if useAttn:
            for di in range(targetLen):
                decOutput, decHidden, decAttn = dec(decInput, decHidden, encOutput)

                outputs[:, di] = decOutput.squeeze(1)

                top= decOutput.argmax(-1)
                decInput = top.squeeze(1).detach()
                
                Attn[di, ...] = decAttn.squeeze(1)

        else:
            for di in range(targetLen):
                decOutput, decHidden = dec(decInput, decHidden)

                outputs[:, di] = decOutput.squeeze(1)

                top= decOutput.argmax(-1)
                decInput = top.squeeze(1).detach()


        outputs = outputs.permute(0, 2, 1)

        loss += criterion(outputs, targetTensor)
        

        AccW = wordAccuracy(outputs, targetTensor)
        AccC = charAccuracy(outputs, targetTensor)
        
        if ret:
            return loss.item(), AccW, AccC, outputs.argmax(dim=1), Attn.permute(1, 0, 2)

        return loss.item(), AccW, AccC

def trainIters(encoder: Encoder, decoder: Decoder, epochs: int, maxLen: int, useAttn: bool, learning_rate=0.001, teacherForcingRatio=teacherForcingRatio,wan=False):
    
    encOptim = Adam(encoder.parameters(), lr=learning_rate)
    decOptim = Adam(decoder.parameters(), lr=learning_rate) 

    criterion = nn.NLLLoss()

    for epoch in range(1, epochs + 1):
        if epoch>2:
            teacherForcingRatio = 0

        loop_obj = tqdm(trainLoader)
        epochTrainAccW, epochTrainAccC,epochTrainLoss = [], [], []
        for training_pair in loop_obj:
            inputTensor = training_pair[0].to(device)
            targetTensor = training_pair[1].to(device)

            loss, accW, accC = train(inputTensor, targetTensor, encoder, decoder, encOptim, decOptim, criterion, maxLen, useAttn, teacherForcingRatio)
            

            loop_obj.set_postfix({f"Loss": f"{loss:0.3f}"})
            loop_obj.set_description(f"Epoch {epoch}")
            
            epochTrainLoss.append(loss)
            epochTrainAccW.append(accW)
            epochTrainAccC.append(accC)

        print(f"Training Loss is {sum(epochTrainLoss)/len(epochTrainLoss)}")        
        print(f"Training Word Accuracy is {sum(epochTrainAccW)/len(epochTrainAccW)}")
        print(f"Training Character Accuracy is {sum(epochTrainAccC)/len(epochTrainAccC)}")

        epochValLoss, epochValAccW, epochValAccC = [],  [], []

        for pair in tqdm(valLoader):
            loss, accW, accC = evaluate(encoder, decoder, pair, criterion, useAttn)

            epochValLoss.append(loss)
            epochValAccW.append(accW)
            epochValAccC.append(accC)


        print(f"Validation Loss is {sum(epochValLoss)/len(epochValLoss)}")        
        print(f"Validation Word Accuracy is {sum(epochValAccW)/len(epochValAccW)}")
        print(f"Validation Character Accuracy is {sum(epochValAccC)/len(epochValAccC)}")

        if wan:
            wandb.log(
                {
                    "train_loss": sum(epochTrainLoss)/len(epochTrainLoss),
                    "train_accuracy": sum(epochTrainAccW)/len(epochTrainAccW),
                    "val_loss": sum(epochValLoss)/len(epochValLoss),
                    "val_accuracy": sum(epochValAccW)/len(epochValAccW)
                }
            ) 

def train_wb():
    run = wandb.init()
    config = wandb.config
    wandb.run.name = "bb_{}_nl_{}_dr_{}_hz_{}_tfr_{}".format(config.backbone, config.numHiddenLayers,\
                            config.dropout, config.hiddenSize, config.teacherForcingRatio)
      
    enc = Encoder(inputSize, config.embedSize, config.hiddenSize, config.numHiddenLayers, config.dropout, config.bidirectional, config.backbone).to(device)
    if args.attention:
        dec = AttentionDecoder(outputSize, config.embedSize, config.hiddenSize, config.numHiddenLayers, config.dropout, config.bidirectional, inputSize, config.backbone).to(device)
    else:
        dec = Decoder(outputSize, config.embedSize, config.hiddenSize, config.numHiddenLayers, config.dropout, config.bidirectional, config.backbone).to(device)

    trainIters(enc, dec, args.epochs, inputSize, args.attention, teacherForcingRatio=config.teacherForcingRatio, wan=True)



if __name__ == "__main__":


    if args.question == 2:
        wandb.login(key="e99813e81e3838e6607d858a20693d589933495f")
        with open("./sweep.yml", "r") as f:
            sweep_config = yaml.safe_load(f)

        sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
        wandb.agent(sweep_id, function=train_wb, count=50)

    elif args.question == 4:
        enc = Encoder(inputSize, args.embedSize, args.hiddenSize, args.numHiddenLayers, args.dropout, args.bidirectional, args.backbone).to(device)
        
        if not args.attention:
            dec = Decoder(outputSize, args.embedSize, args.hiddenSize, args.numHiddenLayers, args.dropout, args.bidirectional, args.backbone).to(device)
            
            trainIters(enc, dec, args.epochs, inputSize, args.attention, teacherForcingRatio=args.teacherForcingRatio, wan=False)
            criterion = nn.NLLLoss()
            
            acc = [];outs = []
            for pair in testLoader:
                _, accuracy, _, ret, _ = evaluate(enc, dec, pair, criterion, args.attention, True)
                acc.append(accuracy)
                outs.append(ret)
            
            print(f"\nWe get {sum(acc)/len(acc)*100}%  Test accuracy.")
            outs = torch.cat(outs, dim=0)
            
            word2csv(outs, testData.y2TDictR, "pred", "./aksharantar_sampled/tam/tam_test.csv")

        else:
            dec = AttentionDecoder(outputSize, args.embedSize, args.hiddenSize, args.numHiddenLayers, args.dropout, args.bidirectional, inputSize, args.backbone).to(device)
            
            trainIters(enc, dec, args.epochs, inputSize, args.attention, teacherForcingRatio=args.teacherForcingRatio, wan=False)
            criterion = nn.NLLLoss()
            
            acc = [];outs = [];attn = []
            for pair in testLoader:
                _, accuracy, _, ret, decAttn = evaluate(enc, dec, pair, criterion, args.attention, True)
                acc.append(accuracy)
                outs.append(ret)

                attn.append(decAttn)
            
            print(f"\nWe get {sum(acc)/len(acc)*100}%  Test accuracy.")
            
            outs = torch.cat(outs, dim=0)
            word2csv(outs, testData.y2TDictR, "predAttn", "./aksharantar_sampled/tam/tam_test.csv")

            attn = torch.cat(attn, dim=0)
            plot(attn, "attnMap")

    else:


        encoder = Encoder(inputSize, args.embedSize, args.hiddenSize, args.numHiddenLayers, args.dropout, args.bidirectional, args.backbone).to(device)
        if args.attention:
            decoder = AttentionDecoder(outputSize, args.embedSize, args.hiddenSize, args.numHiddenLayers, args.dropout, args.bidirectional, inputSize, args.backbone).to(device)
        else:   
            decoder = Decoder(outputSize, args.embedSize, args.hiddenSize, args.numHiddenLayers, args.dropout, args.bidirectional, args.backbone).to(device)

        trainIters(encoder, decoder, args.epochs, inputSize, args.attention, teacherForcingRatio=args.teacherForcingRatio, wan=False)






