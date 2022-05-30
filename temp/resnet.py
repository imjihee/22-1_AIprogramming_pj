import timm
import torch
import torch.nn as n
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle5 as pickle
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torchvision.transforms.functional import to_pil_image
from utils import AverageMeter, simple_accuracy, valid

batch_size = 512
lr = 0.0001
weight_decay = 0.000001
t_total = 10000
eval_every = 400
train = True

with open('../data/Dataset5', 'rb') as handle:
    data = pickle.load(handle)
    y = pickle.load(handle)
    testdata = pickle.load(handle)
    testy = pickle.load(handle)

"""
Data Preprocessing: reshape, interpolate
"""
data = torch.tensor([data],dtype=torch.float)/255
data = data.view(-1,10,10,3).permute(0,3,1,2)
y = torch.tensor([y-1],dtype = torch.long).view(-1)
#data = data.repeat(1,3, 1, 1)

testdata = torch.tensor([testdata],dtype=torch.float)/255
testdata = testdata.view(-1,10,10,3).permute(0,3,1,2)#.unsqueeze(1)
testy = torch.tensor([testy-1],dtype = torch.long).view(-1)
num=1000
#to_pil_image(data[num]).save("save/before_interpolate_"+str(num)+".png", format="png")

data = F.interpolate(data, size=(32, 32), mode='bilinear')
testdata = F.interpolate(testdata, size=(32, 32), mode='bilinear')
#to_pil_image(data[num]).save("save/interpolate_"+str(num)+".png", format="png")

dataset = torch.utils.data.TensorDataset(data, y)
test_dataset = torch.utils.data.TensorDataset(testdata, testy)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""
Train Setting
"""
if train == True:
    train_loader = DataLoader(dataset, batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)

    #create model
    model = timm.create_model('efficientnetv2_rw_s', pretrained=True, num_classes=5)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)

    loss_fct = CrossEntropyLoss()
    losses = AverageMeter()
    global_step = 0
    scaler = torch.cuda.amp.GradScaler()

    #Start Train
    model.train()
    epoch_iterator = tqdm(train_loader,
                          desc="Training (X / X Steps) (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True
                         )

    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(device) for t in batch)
        optimizer.zero_grad()
        x, y = batch
        y = y.view(-1)

        with torch.cuda.amp.autocast():
            logit = model(x)
            #print(logit)
            #print(logit.shape)
            loss = loss_fct(logit,y)
            #print(loss.item())
        losses.update(loss.item())
        #loss.backward()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        #optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)

        scaler.step(optimizer)
        scaler.update()


        global_step += 1
        epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
        #Start Validation
        if global_step % eval_every == 0:
            acc = valid(model, test_loader)