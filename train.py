#imports
#it has a function file named my_functions.py
#this file contains the input parameters for training of the model

#there are 2 models namely - 'densenet121' and 'alexnet'
#Select from these 2 models

# Take gpu as an argument to run train.py


import numpy as np
import pandas as pd
import time
import torch
from torch.autograd import variable
import seaborn as sb
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, transforms
import torchvision.models as models
import time
import argparse
from os import listdir
#from classifier import classifier
from my_functions import do_deep_learning

train_parser = argparse.ArgumentParser()

 # User should be able to type python train.py data_directory
    # Non-optional argument - must be input (not as -- just the direct name i.e. python train.py flowers)
#train_parser.add_argument('--dir', action="store", nargs='*', default="/flowers/")
train_parser.add_argument('data_dir', type = str, nargs='*', default = "/home/workspace/aipnd-project/flowers/")
    # Choose where to save the checkpoint
train_parser.add_argument('--save_dir', action="store", dest="save_dir", default="/home/workspace/paind-project/checkpoint.pth")
    # Choose model architecture
train_parser.add_argument('--arch', action="store", dest="model", default="densenet121")
    # Choose learning rate
train_parser.add_argument('--learning_rate', action="store", dest="learning_rate", default=0.001)
    # Choose number of epochs
train_parser.add_argument('--epochs', action="store", dest="epochs", type=int, default=2)
    # Choose number of hidden units
train_parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=1024)
    # Choose processor
train_parser.add_argument('--processor', action="store", dest="processor", default="GPU")

train_args = train_parser.parse_args()
print("Image Directory: ", train_args.data_dir, ";  Save Directory: ", train_args.save_dir, ";  Model: ", train_args.model, "; Learning Rate: ", train_args.learning_rate, "; Epochs: ", train_args.epochs, "; Hidden units: ", train_args.hidden_units, "; Processor :", train_args.processor)

def main():
    print(train_args.model)
    do_deep_learning(train_args.model, train_args.data_dir, train_args.save_dir, train_args.learning_rate, train_args.epochs, train_args.hidden_units, train_args.processor)

main()

# Call to get_input_args function to run the program
if __name__ == "__main__":
    main()
