# DC-GAN
A Simple implementation of DC-GANs using PyTorch.

A DC-GAN consists of two neural networks working opposite to each other.
As in the case of Game Theory aim is to find a Nash equilbrium where both the
networks find a optimal solution to the optimization problem.

## Discriminator

This is the basic architecture of Discriminator used  in this demo.
<center>
  <img src="/image/Disc.png" align="middle" width="90%" alt= "Discriminator"/>
</center>

## Generator

This is the basic architecture of Generator used  in this demo.
<p>
  <img src="/image/Gen.png" align="middle" width="90%" alt= "Generator"/>
</p>

## Gan Hacks Used.
+ Used Batch-Norm in both Gen and Disc
+ Used Leaky ReLU for Disc and ReLU for Gen
+ Used one sided label Smoothning for real images for training Disc
+ Used more Fake images to train the Gen (Did not have much effect on Results)

# Results on Standard Datasets

## CIFAR-10

### Loss and F1 merics 

<p float="center">
  <img src="/image/DC_32/GAN_loss.png" width="45%" />
  <img src="/image/DC_32/F1_metrics_norm.png" width="45%" /> 
</p>

### Fake Reconstructions

<p float="center">
  <img src="/image/DC_32/train_epoch_99.png" align="middle" width="80%" alt= "Fake Samples"/>
</p>

## STL-10

### Loss and F1 merics 

<p float="left">
  <img src="/image/STL10_64/svg/GAN_loss.svg" width="45%" />
  <img src="/image/STL10_64/svg/F1_metrics_norm.svg" width="45%" /> 
</p>

### Fake Reconstructions

<p float="left">
  <img src="/image/STL10_64/svg/train_epoch_199.png" align="middle" width="80%" alt= "Fake Samples"/>
</p>

## CELEBA

### Loss and F1 merics 

<p float="left">
  <img src="/image/Celeba_64/svg/GAN_loss.svg" width="45%" />
  <img src="/image/Celeba_64/svg/F1_metrics_norm.svg" width="45%" /> 
</p>

### Fake Reconstructions

<p float="left">
  <img src="/image/Celeba_64/svg/train_epoch_115.svg" align="middle" width="80%" alt= "Fake Samples"/>
</p>

# Observations
+ It is clearly visible that in generation of fake images that the generator is able to generate few features very accurately but fails for a class with multiple orientations as in cifar10 and stl10 cases. Class conditional images can be reproduced using C-GAN (will try out later).
+ Also Dropout Layers could be added in the generator and discriminator and also using InstanceNorm instead of BatchNorm as it is found to make average image of all classes with BatchNorm.
+ As per my Observation it is found that the official pytorch DC-GAN implementation does not work that well for Cifar10 and Stl10 datasets.

