# This file contains the process_image and predict function.

import matplotlib.pyplot as plt
import numpy as np
import time
import seaborn as sb
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import argparse
from predict import predict_args
import glob
import PIL
from PIL import Image


def process_image(image):
    img = Image.open(image)

    width, height = img.size
    #print(img.size)
    
    if width < height:
        img.thumbnail((256,height))
    else:
        img.thumbnail((width,256))

    #print(img.size)
    imgwidth = img.size[0]
    imgheight = img.size[1]
    halfimgwidth = imgwidth//2
    halfimgheight = imgheight//2
   
    crop_square = (imgwidth//2 - 112, 
                   imgheight//2 - 112, 
                   imgwidth//2 + 112, 
                   imgheight//2 + 112)

    img = img.crop(crop_square)
    to_tensor = transforms.ToTensor()
    img = to_tensor(img)
    #print(img)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
    img = normalize(img)
    #print(img.shape)
    img = np.array(img)

    #print(img.shape)
    img = np.ndarray.transpose(img)
    #print(img.shape)
    return img

def predict(image_path, model, topk = 5, processor = 'GPU'):
    
    cat = model.class_to_idx
    test_image = process_image(image_path)
    images = torch.from_numpy(test_image)
   
    #image = image.float()
    image1 = np.transpose(images, (2, 1, 0))
    #print(image.shape)
    
    with torch.no_grad():
        model.eval()
        images = image1.unsqueeze_(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images = images.to(device)
        model = model.to(device)
        output = model.forward(images)
    ps = torch.exp(output)
    check = torch.topk(ps, topk)
    probs = check[0].cpu()
    classes = check[1].cpu()
    c = classes.numpy()
    c = list(c)
    l1 = []
    for i in range(topk):
        z = c[0][i]
        for a in cat.keys():
            if cat[a] == z:
                r = a
                l1.append(r)
    return probs, l1