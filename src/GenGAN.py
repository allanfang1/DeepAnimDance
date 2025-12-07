
import numpy as np
import cv2
import os
import pickle
import sys
import math

import matplotlib.pyplot as plt

from torchvision.io import read_image
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton
from GenVanillaNN import * 



class Discriminator(nn.Module):
    def __init__(self, ngpu=0):
        super().__init__()
        self.ngpu = ngpu
        # ALLAN'S IMPLEMENTATION -TODO
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), #32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), #16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), #8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), #4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 1, 4, 1, 0), #1
            nn.Sigmoid(),
        )

    def forward(self, input):
        pass
        return self.model(input)

class GenGAN():
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False):
        # ALLAN'S IMPLEMENTATION (changed to use network ImToImage instead of 26ToImage) -TODO
        # self.netG = GenNNSke26ToImage()
        self.netG = GenNNSkeImToImage()
        self.netD = Discriminator()
        image_size = 64
        self.real_label = 1.
        self.fake_label = 0.
        self.filename = 'data/Dance/DanceGenGAN.pth'
        src_transform = transforms.Compose([ SkeToImageTransform(image_size),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                 ])
        tgt_transform = transforms.Compose(
                            [transforms.Resize((64, 64)),
                            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            transforms.CenterCrop(64),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform, source_transform=src_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=32, shuffle=True)
        if loadFromFile and os.path.isfile(self.filename):
            print("GenGAN: Load=", self.filename, "   Current Working Directory=", os.getcwd())
            self.netG = torch.load(self.filename)


    def train(self, n_epochs=20):
        # ALLAN'S IMPLEMENTATION -TODO (https://arxiv.org/abs/1611.07004)
        criterion_BCE = nn.BCELoss()
        criterion_L1 = nn.L1Loss()
        lambda_L1 = 100.0 # L1 weight
        optimizerG = torch.optim.Adam(self.netG.parameters(), lr=1e-4)
        optimizerD = torch.optim.Adam(self.netD.parameters(), lr=1e-4)

        print(f"Training for {n_epochs} epochs...")
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for i, (skeletons, real_images) in enumerate(self.dataloader):
                # train discriminator
                optimizerD.zero_grad()

                # real images
                output = self.netD(real_images).view(-1)
                labels = torch.ones_like(output)
                lossD_real = criterion_BCE(output, labels)
                lossD_real.backward()

                # fake images
                fake_images = self.netG(skeletons)
                output = self.netD(fake_images.detach()).view(-1)
                labels = torch.zeros_like(output) 
                lossD_fake = criterion_BCE(output, labels)
                lossD_fake.backward()
                optimizerD.step()

                # train generator
                optimizerG.zero_grad()
                output = self.netD(fake_images).view(-1)
                labels = torch.ones_like(output)
                lossG_L1 = criterion_L1(fake_images, real_images)
                lossG_GAN = criterion_BCE(output, labels)
                lossG = lossG_GAN + lambda_L1 * lossG_L1
                lossG.backward()
                optimizerG.step()
                
                epoch_loss += lossG.item()
                num_batches += 1

            # avg_loss = epoch_loss / num_batches
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{n_epochs}], D_Loss: {lossD_real.item() + lossD_fake.item():.4f}, G_Loss_L1: {lossG_L1.item():.4f}, G_Loss_BCE: {lossG_GAN.item():.4f}, G_Loss: {lossG.item():.4f}")
    
        print(f"Saving model to {self.filename}")
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        torch.save(self.netG, self.filename)

    def generate(self, ske):
        """ generator of image from skeleton """
        # ALLAN'S IMPLEMENTATION -TODO
        ske_t = self.dataset.preprocessSkeleton(ske)
        ske_t_batch = ske_t.unsqueeze(0)        # make a batch
        normalized_output = self.netG(ske_t_batch)
        res = self.dataset.tensor2image(normalized_output[0])       # get image 0 from the batch
        return res




if __name__ == '__main__':
    force = False
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/taichi1.mp4"
    print("GenGAN: Current Working Directory=", os.getcwd())
    print("GenGAN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    #if False:
    if True:    # train or load
        # Train
        gen = GenGAN(targetVideoSke, False)
        gen.train(200) #5) #200)
    else:
        gen = GenGAN(targetVideoSke, loadFromFile=True)    # load from file        


    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        #image = image*255
        nouvelle_taille = (256, 256) 
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)

