import time
import torch.nn as nn
#import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch.utils.data import Dataset
import torch, math
import torch.fft
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
# !pip install vit-pytorch
import time
import torch.nn.functional as F
import pywt
from torch.autograd import Function
from torch.utils.data import DataLoader
# from torch.utils.data import Dataset
import torch.optim as optim
# !pip install torchsummary
from torchsummary import summary
# !pip install einops
from math import ceil
import os
import copy
import torchvision.models as models
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import warnings
warnings.filterwarnings("ignore")
# helpers
from einops import reduce
from fhist_test import FhistDataset
from tqdm import tqdm
#from FNET_model import FNet2D
import tensorboard
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# transform = transforms.Compose([transforms.ToTensor(),transforms.Resize([512, 384])])
transform = transforms.Compose([transforms.ToTensor(),transforms.Resize([224, 224]),transforms.ColorJitter(0.15,0.15,0.15,0.1), transforms.RandomHorizontalFlip()])
batch_size = 1600

# train_dataset = bachdataset(path='/home/nilgiri/Desktop/ravi/bach_dataset/train.csv',transforms=transform )
val_dataset = FhistDataset(path='/home/ravi/Domain_adap_code/data_target_test.csv' ,transforms=transform)

# trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
testloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = models.vit_b_16(pretrained=True)
#print(model)
#exit()
model.fc = nn.Linear(in_features=768, out_features=6, bias=True)
model.load_state_dict(torch.load('/home/ravi/Domain_adap_code/tvt_wts_train/FNet.pth'),strict=False)

model.to(device)

criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()


optimizer = optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
best_acc = 0.0
for epoch in range(1):  # loop over the dataset multiple times
    t0 = time.time()
    running_corrects = 0
    running_loss = 0.0
    df=pd.DataFrame(columns=['slide_path','label','Softmax'])
    temp_dict={}

    running_test_loss=0
    running_test_correct=0

    for data in tqdm(testloader):
        images, labels = data[0].to(device), data[1].to(device)
        with torch.no_grad():
            outputs = model(images)
            s_out= F.softmax(outputs)
        _, predicted = torch.max(outputs.data, 1)
        test_loss= criterion(outputs,labels)
        running_test_loss=test_loss.item()*images.size(0)
        running_test_correct+= torch.sum(predicted == labels.data)
       
    epoch_val_loss = running_test_loss / len(val_dataset)
    epoch_val_acc = running_test_correct.double() / len(val_dataset)        
    writer.add_scalar("Loss/Val", epoch_val_loss, epoch+1)
    writer.add_scalar("Acc/Val", epoch_val_acc, epoch+1)
    print(f"Epoch : {epoch+1} - Val_loss : {epoch_val_loss:.4f} - Val_Acc: {epoch_val_acc:.4f}  - Time: {time.time() - t0}\n")

print('Finished Testing')


#     for data in tqdm(testloader):
#         images, labels, path = data[0], data[1], data[2]
#         with torch.no_grad():
#             outputs = model(images.to(device))
#             s_out= F.softmax(outputs)
#             temp_dict['slide_path']=path
#             temp_dict['label']=labels
#             temp_dict['Softmax']=s_out.tolist()
#             df=pd.concat([df,pd.DataFrame(temp_dict)])
           
# print('Finished Testing')
# df.to_csv('sm_resnet_3class.csv',index=False)
