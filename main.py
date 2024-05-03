import os

import cv2
import numpy as np
import torch
from DataLoader import DataLoader
from DataVisualizer import DataVisualizer
from config import BASE_PATH
from train import train, eval
imgfolderpath = BASE_PATH + '/images/val/'
labelfolderpath = BASE_PATH + '/labels/val/'

if __name__ == "__main__":

    loader = DataLoader(imgfolderpath=imgfolderpath, labelfolderpath=labelfolderpath)
    dataset = loader.init_dataset()
    # train()
    eval('wb_localization_dataset/images/val/nlvnpf-0137-01-047.jpg')

    # visualizer = DataVisualizer()
    # visualizer.visualize_data(dataset[1])