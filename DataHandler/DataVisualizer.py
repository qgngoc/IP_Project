import sys
import os
parent_dir = os.path.dirname(os.path.realpath(__file__)) + "/../"
sys.path.append(parent_dir)

import cv2
import numpy as np
import torch

import DataLoader
from config import BASE_PATH
# from matplotlib import pyplot as plt

class DataVisualizer:
    def __init__(self):
        pass

    @staticmethod
    def visualize_data(data):
        image, label = data
        image = image * 255
        label = label[1].tolist()
        image = np.array(image.permute(1,2,0), dtype='uint8')
        # cv2.imshow("Img", image)
        # plt.show(image)
        # # cv2.waitKey(0)
        for box in label:
            x_start, y_start, x_end, y_end = box
            # width = int(width)
            # height = int(height)
            x_start = int(x_start)
            y_start = int(y_start)
            x_end = int(x_end)
            y_end = int(y_end)
            cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

        cv2.imshow("Img", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    imgfolderpath = os.path.dirname(os.path.realpath(__file__)) + "/../" + 'wb_localization_dataset' + '/images/train/'
    labelfolderpath = os.path.dirname(os.path.realpath(__file__)) + "/../" + 'wb_localization_dataset' + '/labels/train/'

    dataloader = DataLoader.DataLoader(imgfolderpath, labelfolderpath)
    train_dataset = dataloader.init_dataset()
    data = train_dataset[0]
    DataVisualizer.visualize_data(data=data)