# DC-GAN
A Simple implementation of DC-GANs using PyTorch.

    A DC-GAN consists of two neural networks working opposite to each other. \n
As in the case of Game Theory aim is to find a Nash equilbrium where both the \n
networks find a optimal solution to the optimization problem.

## Discriminator

This is the basic architecture of Discriminator used  in this demo.
<p float="left">
  <img src="/image/Disc.png" align="middle" width="50%" alt= "Discriminator"/>
</p>

## Generator

This is the basic architecture of Generator used  in this demo.
<p float="left">
  <img src="/image/Gen.png" align="middle" width="50%" alt= "Generator"/>
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
  <img src="/image/DC_32/GAN_loss.png" width="50%" />
  <img src="/image/DC_32/F1_metrics_norm.png" width="50%" /> 
</p>

### Fake Reconstructions

<p float="center">
  <img src="/image/DC_32/train_epoch_99.png" align="middle" width="50%" alt= "Generator"/>
</p>

## STL-10

### Loss and F1 merics 

<p float="left">
  <img src="/image/STL10_64/svg/GAN_loss.svg" width="50%" />
  <img src="/image/STL10_64/svg/F1_metrics_norm.svg" width="50%" /> 
</p>

### Fake Reconstructions

<p float="left">
  <img src="/image/STL10_64/svg/train_epoch_199.png" align="middle" width="50%" alt= "Generator"/>
</p>

## CELEBA

### Loss and F1 merics 

<p float="left">
  <img src="/image/Celeba_64/svg/GAN_loss.svg" width="50%" />
  <img src="/image/Celeba_64/svg/F1_metrics_norm.svg" width="50%" /> 
</p>

### Fake Reconstructions

<p float="left">
  <img src="/image/Celeba_64/svg/train_epoch_115.svg" align="middle" width="50%" alt= "Generator"/>
</p>
