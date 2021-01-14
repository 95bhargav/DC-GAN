# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 12:21:00 2020

@author: Shukla
"""
#changelog
Changes = "Official DC-GAN implementation (with mod networks) \
             \n \t\t\t CelebA img size = 64 one-sided label smoothning\
                \n \t\t\t using Lr Scheduler for Gen and Disc \
                    \n \t\t\t ngf*2, step changes"  #Reason for this run

#%% 1. Importing Libraries

import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import sys
import numpy as np
from sklearn.metrics import f1_score,confusion_matrix

CentralFolder =os.getcwd()
sys.path.append(CentralFolder)
import HelperFunctions as HelpFunc

manualSeed = torch.randint(0,1000,(1,))
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)

torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.enabled = False 
print("Seed     \t: ",manualSeed)
print("Changes  \t: ", Changes)

#%% 2. Folder Creation 

trainpath =os.path.join(CentralFolder,'runs')
time_str=time.strftime("%y_%m_%d__%H_%M_%S")
trainfolder=os.path.join(trainpath,time_str)
if not os.path.exists(trainfolder):
    os.makedirs(trainfolder) 
    
#%%
class Generator_DC_GAN(nn.Module):
    def __init__(self, In ,Out, code, Kernel=4, Stride=2, Pad=1, image_size=64):
        super(Generator_DC_GAN, self).__init__()
        
        self.image_size = image_size
        
        self.main = nn.Sequential(
            
            nn.ConvTranspose2d( code, Out * 8, Kernel, Stride-1, Pad-1,
                                bias=False),
            nn.BatchNorm2d(Out * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(Out * 8, Out * 4, Kernel, Stride, Pad,
                                bias=False),
            nn.BatchNorm2d(Out * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d( Out * 4, Out * 2, Kernel, Stride, Pad,
                                bias=False),
            nn.BatchNorm2d(Out * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d( Out * 2, Out, Kernel, Stride, Pad,
                                bias=False),
            nn.BatchNorm2d(Out),
            nn.ReLU(True),
            
            nn.ConvTranspose2d( Out, Out,1,1,0, bias=False),
            nn.BatchNorm2d(Out),
            nn.ReLU(True))
        
        if self.image_size == 32:
            self.main.add_module("Final_Layer", nn.Sequential(
                nn.ConvTranspose2d(Out, In, 3,1,1, bias = False),
                nn.Tanh())) 
        
        elif self.image_size == 64:
            self.main.add_module("Final_Layer", nn.Sequential(
               nn.ConvTranspose2d(Out, In, Kernel, Stride, Pad, bias=False),
               nn.Tanh())) 

    def forward(self, inp):
        return self.main(inp)

class Discriminator_DC_GAN(nn.Module):
    def __init__(self, nc ,ndf, image_size = 64):
        super(Discriminator_DC_GAN, self).__init__()
        
        self.image_size = image_size
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True))
        
        if self.image_size == 32:
            self.main.add_module("Final_Layer", nn.Sequential(
                nn.Conv2d(ndf * 8, 1, 2, 2, 0, bias=False),
                nn.Sigmoid()))
            
        elif self.image_size == 64:
            self.main.add_module("Final_Layer",nn.Sequential(
                
                nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1,  bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
    
                nn.Conv2d(ndf * 8, 1, 2, 2, 0, bias=False),
                nn.Sigmoid()))

    def forward(self, inp):
        output = self.main(inp)
        return output.view(-1, 1).squeeze(1)

#%% 3. Classes and Functions
   
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

def f1_metrics(labels,preds):
    '''
    Inputs:
        labels : nested list of the labels for discriminator \n
        preds  : nested list of the predictions of discriminator \n
    
    Returns:
        Dict containing F1 metrics for the set \n
    
    '''
    metrics = dict({})
    metrics['tn'] =[]
    metrics['fp'] =[]
    metrics['fn'] =[]
    metrics['tp'] =[]
    metrics['samples']      = []
    metrics['f1_score']     = []
    
    
    for i in range(len(labels)):
        labels1 = labels[i].cpu().detach().numpy()
        preds1  = preds[i].cpu().detach().numpy()
        tn,fp, fn,tp = confusion_matrix(labels1,preds1,labels=[0,1]).ravel()
        f1 = f1_score(labels1,preds1)  
        
        
        metrics['tn'].append(tn/labels[i].shape[0])
        metrics['fp'].append(fp/labels[i].shape[0])
        metrics['fn'].append(fn/labels[i].shape[0])
        metrics['tp'].append(tp/labels[i].shape[0])
        metrics['samples'].append(labels[i].shape[0])
        metrics['f1_score'].append(f1)

    return metrics


def imshow(img,dataset= "cifar10"):
    """
    Unnormalize the image and plot it in figure
    """
    img = img/2 +0.5
    if dataset =='mnist':
        plt.imshow(img.squeeze().cpu().detach(), cmap="gray")
    elif dataset == 'cifar10':
        torchimg =img.permute( 1, 2, 0)
        plt.imshow(torchimg.squeeze().cpu().detach())
        
def PlotViz(fixed_data,FigureDict,PlotTitle, dataset='cifar10'):
    """
    Inputs:
        fixed_data  : Real image to reconstruct. \n
        FigureDict  : Dictionary object to store figures. 'None' if not available.\n
        PlotTitle   : Title for Figure. 'None' if not available.\n
        dataset     : The current working Dataset. \n
        
    Returns:
        Matplotlib figure stored in working directory.
        
    """
    fig_ae  = plt.figure(figsize=(10,10),dpi=300)
    fig_ae.suptitle(PlotTitle)
    
    for i in range(64):
        plt.subplot(8,8,i+1) 
        imshow(fixed_data[i],dataset)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.xticks([])
        plt.yticks([])

    if FigureDict is not None:
        FigureDict.Store(fig=fig_ae, name=PlotTitle, save = True)
    plt.show()


def train(dataloader, fixed_noise,netD, netG, criterion, optimizerD,
          optimizerG, CurrentBatch, epoch, Reporter):
    
    for i, data in enumerate(dataloader):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        optimizerD.zero_grad()
        real_cpu = data[0].to(_DEVICE)
        batch_size = real_cpu.size(0)
        randn_lab = float(torch.randn(1,).uniform_(0.9,1.0))
        label = torch.full((batch_size,), real_label*randn_lab,
                           dtype=real_cpu.dtype, device=_DEVICE)

        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        
        #Accuracy of Disc on real images

        real_preds          = torch.round(output)
        real_lab            = torch.round(label)
        correct_real        = (real_preds == real_lab).sum().float().item()
        d_accuracy_real     = (correct_real*100/real_lab.size(0))

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=_DEVICE)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        
        #Accuracy of Disc on fake images

        fake_preds          = torch.round(output)
        fake_lab            = torch.round(label)
        correct_fake        = (fake_preds == fake_lab).sum().float().item()
        d_accuracy_fake     = (correct_fake*100/fake_lab.size(0))

        #Overall Accuracy of discriminator
        d_accuracy = (correct_real+correct_fake)*100/(real_lab.size(0)+fake_lab.size(0))       
        
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        
        optimizerG.zero_grad()
        output = netD(fake)
        label.fill_(real_label)            # fake labels are real for generator cost
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()
        

        #Appending the labels and predictions to list to use for F1 metrics
        labs  = torch.cat((real_lab,fake_lab))
        preds = torch.cat((real_preds,fake_preds))
        
        Reporter.DumpValues['Disc_labels'].append(labs)
        Reporter.DumpValues['Disc_preds'].append(preds)
        Reporter.DumpValues['Lr_disc'].append(optimizerD.param_groups[0]['lr'])
        Reporter.DumpValues['Lr_gen'].append(optimizerG.param_groups[0]['lr'])
        
        
        Reporter.Store([epoch+1,            Reporter.Batch,
                        errD.item(),        errG.item(),
                        errD_real.item(),   errD_fake.item(),
                        d_accuracy,         d_accuracy_real,
                        d_accuracy_fake])
        
        #plotting images side by side
        if Reporter.Batch % len(dataloader) == 0:
            with torch.no_grad():
                fake_img = netG(fixed_noise)
            PlotTitle = "train_epoch_"+str(epoch)
            FigureDict = HelpFunc.FigureStorage(os.path.join(trainfolder,"train_plots_DC_GAN"),
                                            dpi = 300, autosave = True)
            PlotViz(fake_img,FigureDict, PlotTitle)
        
        CurrentBatch+=1
    return CurrentBatch
            

#%% 4. Parameter Initializations
ParameterFile=HelpFunc.ParameterStorage(trainfolder)

global _DEVICE
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

nc=3
nz = 100
ngf = 64*2
ndf = 64
image_size =64
n_epochs =200
Mean = 0.5
Std = 0.5
Lr = 0.0002
Batch_Size = 128
image_size = 64
MileStonesD = [100]
MileStonesG = [100,150]
gammas     = 0.1
dataset = "celeba"


## STL-10 dataset
if dataset == "stl10":
    datapath = os.path.join(os.path.dirname(CentralFolder), "Datasets")
    trainloader, testloader = HelpFunc.LoadSTL10(path =datapath,
                                                 minibatch = Batch_Size,
                                                 normalization= "-11",
                                                 train_split="train+unlabeled",
                                                 image_size=image_size)
elif dataset == "cifar10":
    datapath = os.path.join(os.path.dirname(CentralFolder), "Datasets")
    trainloader, testloader = HelpFunc.LoadCifar10(path = datapath,
                                                   minibatch = Batch_Size,
                                                   normalization= "-11",
                                                   image_size=image_size)

elif dataset == "celeba":
    datapath = os.path.join(os.path.dirname(CentralFolder), "Datasets")
    trainloader, testloader = HelpFunc.LoadCelebaA(path = datapath,
                                                   minibatch = Batch_Size,
                                                   normalization ="-11",
                                                   image_size= image_size)


print("len of dataset : ", len(trainloader)*Batch_Size, "\n")


netG = Generator_DC_GAN(nc,ngf,nz,image_size= image_size).to(_DEVICE)
netG.apply(weights_init)

netD = Discriminator_DC_GAN(nc,ndf,image_size= image_size).to(_DEVICE)
netD.apply(weights_init)

criterion = nn.BCELoss()

fixed_noise = torch.randn(Batch_Size, nz, 1, 1, device=_DEVICE)
real_label = 1.
fake_label = 0.

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=Lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=Lr, betas=(0.5, 0.999))
schedD      = optim.lr_scheduler.MultiStepLR(optimizerD, milestones=MileStonesD, gamma= gammas)
schedG      = optim.lr_scheduler.MultiStepLR(optimizerG, milestones=MileStonesG, gamma= gammas)

#%% 5. Saving

ParameterFile._Create()
ParameterFile.WriteTab("Changes         : ", Changes)
ParameterFile.EqualSigns()

ParameterFile.WriteTab("seed            : ", manualSeed)
ParameterFile.WriteTab("Dataset         : ", dataset)
ParameterFile.WriteTab("Channels        : ", nc)
ParameterFile.WriteTab("Image_size      : ", image_size)
ParameterFile.WriteTab("Latent Dim      : ", nz)
ParameterFile.WriteTab("gen_channels    : ", ngf)
ParameterFile.WriteTab("disc_channels   : ", ndf)
ParameterFile.WriteTab("Epochs          : ", n_epochs)
ParameterFile.WriteTab("Mean            : ", Mean)
ParameterFile.WriteTab("Std             : ", Std)
ParameterFile.WriteTab("Lr              : ", Lr)
ParameterFile.WriteTab("Batch Size      : ", Batch_Size)
ParameterFile.WriteTab("Scheduler StepsD: ", MileStonesD)
ParameterFile.WriteTab("Scheduler StepsG: ", MileStonesG)
ParameterFile.WriteTab("Scheduler Gamma : ", gammas)
ParameterFile.WriteTab("Beta1           : ", 0.5)
ParameterFile.WriteTab("Beta2           : ", 0.999)

ParameterFile.EqualSigns()
ParameterFile.WriteTab("Discriminator   : ", netD)

ParameterFile.EqualSigns()
ParameterFile.WriteTab("Generator       : ", netG)

#%% 6. Training
ReporterNames=['Epoch',"Batch", "LossD","LossG","LossD_real", "LossD_fake",
               'D_Accuracy',"D_Acc_real", "D_Acc_fake"]
    
Reporter = HelpFunc.DataStorage(ReporterNames)


Reporter.DumpValues['Disc_labels']           = []
Reporter.DumpValues['Disc_preds']            = []

Reporter.DumpValues["Lr_disc"]               = []
Reporter.DumpValues["Lr_gen"]                = []
Reporter.Batch         = 0
epoch =0
for epoch in range(n_epochs):
    CurrentBatch=train(trainloader, fixed_noise,netD, netG, criterion, 
                       optimizerD, optimizerG, Reporter.Batch, epoch,
                       Reporter)
    schedD.step()
    schedG.step()

#%% 7. Plotting

print("plotting....")
FigureDict = HelpFunc.FigureStorage(os.path.join(trainfolder,"GAN_plots"),
                                    dpi =300,autosave =True)
############################

disc_lr     = HelpFunc.MovingAverage(Reporter.DumpValues['Lr_disc'],window=n_epochs)
gen_lr      = HelpFunc.MovingAverage(Reporter.DumpValues['Lr_gen'],window=n_epochs)

FigureLoss=plt.figure(figsize=(8,4))
plt.plot(disc_lr, label='Disc_lr')
plt.plot(gen_lr, label='Gen_lr')
plt.xlabel('Steps')
plt.ylabel("Loss")
plt.legend(loc=1)
plt.ylim(bottom =0)
plt.title("Standard DC GAN")
plt.show()
FigureDict.Store(fig=FigureLoss, name="LR_vs_Epoch", save=True)


############################

disc_loss= HelpFunc.MovingAverage(Reporter.StoredValues['LossD'],window=n_epochs)
gen_loss= HelpFunc.MovingAverage(Reporter.StoredValues['LossG'],window=n_epochs)

FigureLoss1=plt.figure(figsize=(8,4))
plt.plot(disc_loss, label='LossD')
plt.plot(gen_loss, label='LossG')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title("Standard DC GAN")
plt.ylim(bottom =0)
plt.legend(loc=1)
plt.show()
FigureDict.Store(fig=FigureLoss1, name="GAN_loss", save=True)

############################
discr_loss= HelpFunc.MovingAverage(Reporter.StoredValues['LossD_real'],window=n_epochs)
discf_loss= HelpFunc.MovingAverage(Reporter.StoredValues['LossD_fake'],window=n_epochs)

FigureLoss=plt.figure(figsize=(8,4))
plt.plot(discr_loss, label='Loss_DReal')
plt.plot(discf_loss, label='Loss_DFake')
plt.ylabel('Loss')
plt.xlabel('Steps')
plt.title("Standard DC GAN")
plt.ylim(bottom =0 ,top = 1.5)
plt.legend(loc=1)
plt.ylim(bottom =0, top =2.5 )
plt.show()
FigureDict.Store(fig=FigureLoss, name="Disc_loss", save=True)

############################
disc_acc= HelpFunc.MovingAverage(Reporter.StoredValues['D_Accuracy'],window=n_epochs)
Figureacc=plt.figure(figsize=(8,4))
plt.plot(disc_acc, label='Acc_D')
plt.legend(loc=4)
plt.xlabel('Steps')
plt.ylabel('Accuracy(%)')
plt.title("Discriminator Accuracy")
plt.ylim(bottom =0, top =105 )
plt.minorticks_on()
plt.show()
FigureDict.Store(fig=Figureacc, name="Disc_Accuracy", save=True)

############################
disc_acc_r= HelpFunc.MovingAverage(Reporter.StoredValues['D_Acc_real'],window=n_epochs)
disc_acc_f= HelpFunc.MovingAverage(Reporter.StoredValues['D_Acc_fake'],window=n_epochs)

Figureacc=plt.figure(figsize=(8,4))
plt.plot(disc_acc_r, label='Acc_DReal')
plt.plot(disc_acc_f, label='Acc_DFake')
plt.legend(loc=4)
plt.xlabel('Steps')
plt.ylabel('Accuracy(%)')
plt.title("Standard DC GAN")
plt.ylim(bottom= 0, top = 105)
plt.minorticks_on()
plt.show()
FigureDict.Store(fig=Figureacc, name="Accuracy", save=True)

############################
metrics = f1_metrics(Reporter.DumpValues['Disc_labels'],
                     Reporter.DumpValues['Disc_preds'])
tn= HelpFunc.MovingAverage(metrics['tn'],window=500)
fp= HelpFunc.MovingAverage(metrics['fp'],window=500)
fn= HelpFunc.MovingAverage(metrics['fn'],window=500)
tp= HelpFunc.MovingAverage(metrics['tp'],window=500)
f1= HelpFunc.MovingAverage(metrics['f1_score'],window=500)


Figuremetrics =plt.figure(figsize=(8,4),dpi =300)
ax = Figuremetrics.add_axes([0.1, 0.1, .7, .8])
ax.plot(tn, '-r', label = 'TN')
ax.plot(fp, '-g', label = 'FP')
ax.plot(fn, '-b', label = 'FN')
ax.plot(tp, '-c', label = 'TP')
ax.plot(f1, '-k', label = 'F1_Score')
ax.set_xlabel('Steps')
ax.set_ylabel('Score')
ax.legend(bbox_to_anchor=(1.05, 1.0),title ='Metrics', loc='upper left', borderaxespad=0.)
ax.grid()
plt.suptitle('F1 metrics of Standard Disc')
plt.gcf().subplots_adjust(left=0.3, right=1.9, bottom=0.8, top=0.9)
plt.ylim(top=1.1)
plt.xlim(left =0)
plt.show()

FigureDict.Store(fig=Figuremetrics, name="F1_metrics_norm", save=True)


#%% 8. Deleting networks
del Reporter, netD, netG, FigureDict, ParameterFile, trainloader, testloader
