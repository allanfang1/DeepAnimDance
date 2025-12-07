
import numpy as np
import cv2
import os
import pickle
import sys
import math

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton



class GenNeirest:
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
       Neirest neighbor method: it select the image in videoSke that has the skeleton closest to the skeleton
    """
    def __init__(self, videoSkeTgt):
        self.videoSkeletonTarget = videoSkeTgt

    def generate(self, ske):           
        """ generator of image from skeleton """
        # ALLAN'S IMPLEMENTATION -TODO
        best_image_index = -1
        best_dist = float('inf')

        for i in range (self.videoSkeletonTarget.skeCount()):
            target_ske = self.videoSkeletonTarget.ske[i]
            dist = ske.distance(target_ske)

            if dist < best_dist:
                best_dist = dist
                best_image_index = i
        
        if best_image_index != -1:
            return self.videoSkeletonTarget.readImage(best_image_index)/255

        empty = np.ones((64,64, 3), dtype=np.uint8)
        return empty




