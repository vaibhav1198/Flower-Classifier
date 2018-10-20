#this is the file containing the function to train the selected model namely 'densenet121' or 'alexnet'

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
from os import listdir



# Implement a function for the validation pass
def validation(model, validloader, criterion):
    test_loss = 0
    accuracy = 0
    for images, labels in validloader:

        images, labels = images.to('cuda'), labels.to('cuda')
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy


def do_deep_learning(model, data_dir, save_dir, learning_rate, num_epochs, hidden_units, processor):
    if model == 'densenet121':
        model = models.densenet121(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False

        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
                                 ('fc1', nn.Linear(1024, 512)),
                                 ('relu1', nn.ReLU()),
                                 ('fc2', nn.Linear(512, 102)) ,
                                 ('output', nn.LogSoftmax(dim = 1))]))

        model.classifier = classifier

    elif model == 'alexnet':
        model = models.alexnet(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
                                 ('fc1', nn.Linear(9216, hidden_units)),
                                 ('relu1', nn.ReLU()),
                                 ('fc2', nn.Linear(hidden_units, 102)) ,
                                 ('output', nn.LogSoftmax(dim = 1))]))

        model.classifier = classifier
        
    else:
        print("Please select from densenet121 or alexnet only")

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    train_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      #transforms.RandomVerticalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
    valid_transform = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    trainset = datasets.ImageFolder(train_dir, transform = train_transform)
    validset = datasets.ImageFolder(valid_dir, transform = valid_transform)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(validset, batch_size = 32, shuffle = True)
    
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    epochs = num_epochs
    print_every = 42
    steps = 0

    # change to cuda
    model.cuda()

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader, criterion)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(validloader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(validloader)))
                running_loss = 0
    # TODO: Save the checkpoint
    checkpoint = {'state_dict': model.state_dict(),
                  'optimizer' : optimizer.state_dict,
                  'classIndex' : trainset.class_to_idx,
                  'epoch' : 5,
                  'hidden' : hidden_units,
                  'model' : model}

    torch.save(checkpoint, save_dir)
    print('\nModel Saved')

    return model
