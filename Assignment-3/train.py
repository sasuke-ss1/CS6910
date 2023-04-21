from dataset import Dataset
from Model import *
import torch
import torch.nn as nn
from torch.optim import Adam
import random
from tqdm import tqdm
import sys
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--path", "-p", type=str, default="aksharantar_sampled", help="The path to the root directory of the dataset.")
parser.add_argument("--learningRate", "-lr", type=float, default=1e-3, help="Learning rate to train the model")
parser.add_argument("--backbone", "-bb", type=str, default="lstm", help="The reccurent model used for encoder and decoder.")
parser.add_argument("--hiddenSize", "-hs", type=int, default=256, help="Dimension of the hidden layer of the backbone.")
parser.add_argument("--nIters", "-nIters", default=7500, type=int, help="Number of iteration to train the model.")
parser.add_argument("--teacherForcingRatio", "-tf", type=float, default=1.0, help="The probabily of using teach forcing training.")
parser.add_argument("--attention", "-a", type=lambda x: (str(x).lower() == 'true'), default=False, help="Flag for using attention in the decoder.")
parser.add_argument("--numHiddenLayers", "-nhl", type=int, default=1, help="Vertical depth of the Encoder")
parser.add_argument("--bidirectional", "-bi", type=lambda x: (str(x).lower() == 'true'), default=False, help="Flag for training bi-directional.")
parser.add_argument("--dropout", "-d", type=float, default=0.0, help="Probability of dropout.")
parser.add_argument("--language", "-l", type=str, default="hin", help="The langugae of the dataset.")
#parser.add_argument("--beamS")

args = parser.parse_args()
path = args.path
teacherForcingRatio = args.teacherForcingRatio


data = Dataset(path, lang=args.language)
trainx, trainxx, trainy = data.get_data("train")
pairs = list(zip(trainx, trainxx))
sowToken = data.x2TDict["\t"] #hardcoded
eowToken = data.x2TDict["\n"] #hardcoded

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(inputTensor: torch.Tensor, targetTensor: torch.Tensor,\
            encoder: Encoder, decoder: Decoder, \
            encOptim: torch.optim, decOptim: torch.optim, \
            criterion: nn.Module, maxLen: int):

    encHidden = encoder.initHidden(device)

    encOptim.zero_grad()
    decOptim.zero_grad()

    inputLen = inputTensor.size(0)
    targetLen = targetTensor.size(0)

    encOutputs = torch.zeros(maxLen, encoder.hiddenSize, device=device)    

    loss = 0

    for ei in range(inputLen):
        encOutput, encHidden = encoder(inputTensor[ei], encHidden)
        #encOutputs[ei] = encOutput[0, 0]
    decInput = torch.tensor([[sowToken]], device=device)

    decHidden = encHidden

    useTeacherForcing = True if random.random() < teacherForcingRatio else False

    if useTeacherForcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(targetLen):
            decOutput, decHidden = decoder(decInput, decHidden)

            loss += criterion(decOutput, targetTensor[di].unsqueeze(0))
            decInput = targetTensor[di].unsqueeze(0)  # Teacher forcing
        
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(targetLen):
            decOutput, decHidden = decoder(decoder_input, decHidden, encOutputs)
            topv, topi = decOutput.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decOutput, targetTensor[di])
            if decoder_input.item() == eowToken:
                break

    loss.backward()

    encOptim.step()
    decOptim.step()

    return loss.item() / targetLen


def trainIters(encoder: Encoder, decoder: Decoder, nIters: int, maxLen,print_every=100, learning_rate=0.001):
    print_loss_total = 0  # Reset every print_every
    

    encOptim = Adam(encoder.parameters(), lr=learning_rate)
    decOptim = Adam(decoder.parameters(), lr=learning_rate) 

    trainPair = [random.choice(pairs) for i in range(nIters)]
    criterion = nn.NLLLoss()

    for iter in tqdm(range(1, nIters + 1)):
        training_pair = trainPair[iter - 1]
        inputTensor = training_pair[0].to(device)
        targetTensor = training_pair[1].to(device)

        loss = train(inputTensor, targetTensor, encoder, decoder, encOptim, decOptim, criterion, maxLen)
        print_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(print_loss_avg)


hiddenSize = args.hiddenSize
nIters = args.nIters
backbone = args.backbone

inputSize = data.xLen
outputSize = data.yLen

encoder = Encoder(inputSize, hiddenSize, args.numHiddenLayers, args.dropout, backbone).to(device)
decoder = Decoder(outputSize, hiddenSize, args.numHiddenLayers, args.dropout, backbone).to(device)

trainIters(encoder, decoder, nIters, inputSize)








